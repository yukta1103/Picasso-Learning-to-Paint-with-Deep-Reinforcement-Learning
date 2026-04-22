# SAC agent for Learning to Paint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

from DRL.actor_sac  import ActorSAC
from DRL.critic_sac import TwinCritic
from DRL.rpm        import rpm
from DRL.wgan       import update as update_wgan, cal_reward, load_gan, save_gan
from DRL.ddpg       import decode as renderer_decode
from utils.util     import soft_update, to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")

# Hyperparameters 
GAMMA        = 0.774   # 0.95**5
TAU          = 0.005
ACTOR_LR     = 3e-4
CRITIC_LR    = 3e-4
ALPHA_LR     = 1e-4
BATCH_SIZE   = 96
WGAN_BATCH   = 96    # Match BATCH_SIZE (96) like DDPG
WGAN_FREQ    = 1     # Update discriminator every step like DDPG
RMSIZE       = 16000   # 400 * 40, matches TD3 setup
UPDATE_FREQ  = 1
WARMUP       = 400


class SAC:
    def __init__(self,
                 env_batch,
                 max_step,
                 action_dim  = 65,
                 writer      = None,
                 resume      = None,
                 output_path = None):

        self.env_batch   = env_batch
        self.max_step    = max_step
        self.action_dim  = action_dim
        self.writer      = writer
        self.output_path = output_path
        self.log         = 0
        self.total_steps = 0

        # networks 
        self.actor         = ActorSAC(num_inputs=9, depth=18, action_dim=action_dim).to(device)
        self.critic        = TwinCritic(num_inputs=12, depth=18).to(device)
        self.critic_target = TwinCritic(num_inputs=12, depth=18).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim  = Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)

        # manual entropy tuning (Exponential Decay) 
        self.alpha = 0.05 

        # replay buffer
        self.memory = rpm(RMSIZE)

        # CoordConv grid
        coord = torch.zeros([1, 2, 128, 128])
        for i in range(128):
            for j in range(128):
                coord[0, 0, i, j] = i / 127.
                coord[0, 1, i, j] = j / 127.
        self.coord = coord.to(device)

        if resume is not None:
            self._load(resume)

    # Unpack a uint8 state tensor into named float components
    def _unpack_state(self, state_t):
        B = state_t.shape[0]
        if state_t.dtype == torch.uint8:
            canvas  = state_t[:, :3].float() / 255.0
            gt      = state_t[:, 3:6].float() / 255.0
            stepnum = state_t[:, 6:7].float() / self.max_step
        else:
            canvas  = state_t[:, :3]
            gt      = state_t[:, 3:6]
            stepnum = state_t[:, 6:7]
        coord = self.coord.expand(B, 2, 128, 128)
        return canvas, gt, stepnum, coord

    def _actor_input(self, canvas, gt, stepnum, coord):
        return torch.cat([canvas, gt, stepnum, coord], dim=1)

    # Action selection
    @torch.no_grad()
    def select_action(self, state, evaluate=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device)
        else:
            state = state.to(device)

        canvas, gt, stepnum, coord = self._unpack_state(state)
        actor_in = self._actor_input(canvas, gt, stepnum, coord)

        if evaluate:
            action = self.actor.act(actor_in)
        else:
            action, _ = self.actor(actor_in)

        return to_numpy(action)

    # Store transition
    def observe(self, state, action, reward, next_state, done):
        s  = torch.tensor(state,      dtype=torch.uint8)
        a  = torch.tensor(action,     dtype=torch.float32)
        r  = torch.tensor(reward,     dtype=torch.float32)
        s2 = torch.tensor(next_state, dtype=torch.uint8)
        d  = torch.tensor(done,       dtype=torch.float32)

        for i in range(self.env_batch):
            self.memory.append([s[i], a[i], r[i], s2[i], d[i]])

        self.total_steps += self.env_batch

    # SAC update
    def update(self, lr=None):
        if lr is not None:
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = lr[0]
            for param_group in self.actor_optim.param_groups:
                param_group['lr'] = lr[1]

        if self.memory.size() < BATCH_SIZE:
            return None, None, None

        states, actions, rewards, next_states, dones = \
            self.memory.sample_batch(BATCH_SIZE, device)

        canvas0,  gt,  stepnum,  coord  = self._unpack_state(states)
        canvas0n, gtn, stepnumn, coordn = self._unpack_state(next_states)

        # stepnum after action applied — matches DDPG's (T+1)/max_step
        stepnum_post = (stepnum * self.max_step + 1).clamp(0, self.max_step) / self.max_step

        with torch.no_grad():
            canvas1 = renderer_decode(actions, canvas0)
        if self.log % WGAN_FREQ == 0:
            idx = torch.randperm(BATCH_SIZE, device=device)[:WGAN_BATCH]
            wgan_fake, wgan_real, wgan_pen = update_wgan(
                canvas1[idx].detach(), gt[idx].detach()
            )
            if self.writer and self.log % 100 == 0:
                self.writer.add_scalar('train/wgan_fake', wgan_fake, self.log)
                self.writer.add_scalar('train/wgan_real', wgan_real, self.log)
                self.writer.add_scalar('train/wgan_pen',  wgan_pen,  self.log)

        # Critic update 
        with torch.no_grad():
            next_actor_in        = self._actor_input(canvas0n, gtn, stepnumn, coordn)
            next_action, next_lp = self.actor(next_actor_in)
            canvas1n             = renderer_decode(next_action, canvas0n)

            q1_next, q2_next = self.critic_target(canvas0n, canvas1n, gtn, stepnumn, coordn)
            
            gan_next = cal_reward(canvas1n, gtn) - cal_reward(canvas0n, gtn)
            
            q_next   = gan_next + torch.min(q1_next, q2_next) - self.alpha * next_lp.unsqueeze(-1)
            
            q_target = GAMMA * (1 - dones.unsqueeze(-1)) * q_next
            
            gan_cur = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)

        q1, q2      = self.critic(canvas0, canvas1.detach(), gt, stepnum_post, coord)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Actor update 
        actor_in             = self._actor_input(canvas0, gt, stepnum, coord)
        new_action, log_prob = self.actor(actor_in)
        canvas1_new          = renderer_decode(new_action, canvas0.detach())

        # Calculate differentiable GAN reward (Model-based backpropagation)
        gan_reward = cal_reward(canvas1_new, gt.detach()) - cal_reward(canvas0.detach(), gt.detach())

        # GAN signal reaches actor through both Q values and direct backpropagation
        q1_new = self.critic.q1_only(
            canvas0.detach(), canvas1_new, gt.detach(),
            stepnum_post.detach(), coord.detach()
        )
        actor_loss = (self.alpha * log_prob - (q1_new.squeeze(-1) + gan_reward.squeeze(-1))).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        #  Alpha update (Manual Decay to break the invisible wall)
        self.alpha = max(0.000001, self.alpha * 0.9995)

        # Soft target update
        soft_update(self.critic_target, self.critic, TAU)

        # Logging 
        if self.writer and self.log % 100 == 0:
            self.writer.add_scalar('train/critic_loss', critic_loss.item(), self.log)
            self.writer.add_scalar('train/actor_loss',  actor_loss.item(),  self.log)
            self.writer.add_scalar('train/alpha',       self.alpha,         self.log)
            self.writer.add_scalar('train/log_prob',    log_prob.mean().item(), self.log)

        del canvas1, canvas1n, canvas1_new
        del q1, q2, q_target, new_action, log_prob, q1_new
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.log += 1
        return actor_loss.item(), critic_loss.item(), self.alpha

    # Persistence
    def save_model(self, path=None):
        path = path or self.output_path
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),  f'{path}/actor_sac.pkl')
        torch.save(self.critic.state_dict(), f'{path}/critic_sac.pkl')
        torch.save({'alpha': self.alpha}, f'{path}/alpha_sac.pkl')
        save_gan(path)
        self.actor.to(device)
        self.critic.to(device)

    def _load(self, path):
        self.actor.load_state_dict( torch.load(f'{path}/actor_sac.pkl',  map_location=device))
        self.critic.load_state_dict(torch.load(f'{path}/critic_sac.pkl', map_location=device))
        if os.path.exists(f'{path}/alpha_sac.pkl'):
            self.alpha = torch.load(f'{path}/alpha_sac.pkl', map_location=device)['alpha']
        if os.path.exists(f'{path}/wgan.pkl'):
            load_gan(path)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()