import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *
from torch.amp import autocast, GradScaler

grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, 1, 128),
    torch.linspace(0, 1, 128),
    indexing='ij'
)
coord = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).to(device)

criterion = nn.MSELoss()

Decoder = FCN()
Decoder.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '../../renderer.pkl'),
    map_location=device
))
Decoder = Decoder.to(device)
Decoder.requires_grad_(False)
Decoder.eval()


def decode(x, canvas):
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


class TD3(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40,
                 tau=0.005, discount=0.9, rmsize=800,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size

        self.actor = ResNet(9, 18, 65)
        self.actor_target = ResNet(9, 18, 65)
        self.critic1 = ResNet_wobn(3 + 9, 18, 1)
        self.critic2 = ResNet_wobn(3 + 9, 18, 1)
        self.critic1_target = ResNet_wobn(3 + 9, 18, 1)
        self.critic2_target = ResNet_wobn(3 + 9, 18, 1)

        target_netD.requires_grad_(False)
        target_netD.eval()

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=1e-2)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=1e-2)

        if resume is not None:
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        self.memory = rpm(rmsize * max_step)

        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.writer = writer
        self.log = 0
        self.total_it = 0

        self.state = None  
        self.action = [None] * self.env_batch
        self.scaler = GradScaler('cuda')
        self.choose_device()

        if hasattr(torch, 'compile') and os.name != 'nt':
            try:
                self.actor = torch.compile(self.actor)
                self.critic1 = torch.compile(self.critic1)
                self.critic2 = torch.compile(self.critic2)
                print(">>> Torch.compile applied successfully.")
            except Exception as e:
                print(f">>> Torch.compile skipped: {e}")

    def play(self, state, target=False):
        state = state.to(device)
        state = torch.cat((
            state[:, :6].float() / 255,
            state[:, 6:7].float() / self.max_step,
            coord.expand(state.shape[0], 2, 128, 128)
        ), 1)
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3:6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)

    def evaluate(self, state, action, target=False, use_critic2=False):
        state = state.to(device)
        action = action.to(device)
        T = state[:, 6:7]
        gt = state[:, 3:6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)

        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt,
                                  (T + 1).float() / self.max_step, coord_], 1)

        if target:
            Q1 = self.critic1_target(merged_state)
            Q2 = self.critic2_target(merged_state)
            Q = torch.min(Q1, Q2)
            return (Q + gan_reward), gan_reward
        else:
            if use_critic2:
                Q = self.critic2(merged_state)
            else:
                Q = self.critic1(merged_state)
            if self.log % 20 == 0 and not use_critic2:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward

    def update_policy(self, lr):
        self.log += 1
        self.total_it += 1

        for param_group in self.critic1_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.critic2_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        state, action, reward, next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.play(next_state, target=True)
            noise = (torch.randn_like(next_action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(0, 1)

            target_q, _ = self.evaluate(next_state, next_action, target=True)

            target_q = reward.to(device) + self.discount * (
                (1 - terminal.float()).view(-1, 1)
            ) * target_q

        with autocast('cuda'):
            cur_q1, step_reward1 = self.evaluate(state, action, use_critic2=False)
            cur_q2, step_reward2 = self.evaluate(state, action, use_critic2=True)
            target_q1 = target_q + step_reward1.detach()
            target_q2 = target_q + step_reward2.detach()
            value_loss1 = criterion(cur_q1, target_q1.detach())
            value_loss2 = criterion(cur_q2, target_q2.detach())
            value_loss = value_loss1 + value_loss2

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.critic1_optim)
        self.scaler.step(self.critic2_optim)

        policy_loss = torch.tensor(0.0, device=device)

        if self.total_it % self.policy_delay == 0:
            with autocast('cuda'):
                actor_action = self.play(state)
                pre_q, _ = self.evaluate(state.detach(), actor_action)
                policy_loss = -pre_q.mean()

            self.actor_optim.zero_grad()
            self.scaler.scale(policy_loss).backward()
            self.scaler.step(self.actor_optim)

            soft_update(self.actor_target, self.actor, self.tau)

        self.scaler.update()

        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

        if self.log % 20 == 0:
            self.writer.add_scalar('train/critic1_loss', value_loss1.item(), self.log)
            self.writer.add_scalar('train/critic2_loss', value_loss2.item(), self.log)
            if self.total_it % self.policy_delay == 0:
                self.writer.add_scalar('train/policy_loss', policy_loss.item(), self.log)

        return -policy_loss, (value_loss1 + value_loss2) / 2

    def observe(self, reward, state, done, step):
        s0 = self.state.detach().cpu()
        a = torch.from_numpy(self.action)
        r = torch.from_numpy(reward.copy())        
        s1 = state.detach().cpu()
        d = torch.from_numpy(done.astype('float32').copy())

        self.memory.append_batch(s0, a, r, s1, d)
        self.state = state

    def noise_action(self, noise_factor, state, action):
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(
                0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs  
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None:
            return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic1.load_state_dict(torch.load('{}/critic1.pkl'.format(path)))
        self.critic2.load_state_dict(torch.load('{}/critic2.pkl'.format(path)))
        load_gan(path)

    def save_model(self, path):
        self.actor.cpu()
        self.critic1.cpu()
        self.critic2.cpu()
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(path))
        torch.save(self.critic1.state_dict(), '{}/critic1.pkl'.format(path))
        torch.save(self.critic2.state_dict(), '{}/critic2.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()

    def choose_device(self):
        Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic1.to(device)
        self.critic1_target.to(device)
        self.critic2.to(device)
        self.critic2_target.to(device)