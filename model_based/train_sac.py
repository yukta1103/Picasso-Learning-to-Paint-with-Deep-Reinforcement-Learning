# SAC training loop for Learning to Paint

import os
import time
import random
import argparse
import numpy as np
import torch

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from utils.util        import get_output_folder
from utils.tensorboard import TensorBoard
from DRL.sac           import SAC, WARMUP, UPDATE_FREQ
from DRL.multi         import fastenv

# Args
parser = argparse.ArgumentParser(description='Learning to Paint — SAC')
parser.add_argument('--max_step',          default=40,      type=int)
parser.add_argument('--env_batch',         default=96,      type=int)
parser.add_argument('--episode_train_times', default=10,    type=int)
parser.add_argument('--train_steps',       default=2000000, type=int) 
parser.add_argument('--validate_interval', default=50,      type=int) 
parser.add_argument('--validate_episodes', default=5,       type=int)
parser.add_argument('--resume',            default=None,    type=str)
parser.add_argument('--resume_step',       default=0,       type=int)
parser.add_argument('--output',            default='./model', type=str)
parser.add_argument('--seed',              default=1234,    type=int)
parser.add_argument('--debug',             action='store_true')
args = parser.parse_args()

# Seeding
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

# Logging
exp    = os.path.abspath('.').split('/')[-1]
writer = TensorBoard('../train_log/{}_sac'.format(exp))
os.system('ln -sf ../train_log/{}_sac ./log_sac'.format(exp))
os.system('mkdir -p ./model')
args.output = get_output_folder(args.output, 'Paint-sac')

# Environment + Agent
env = fastenv(max_episode_length=args.max_step,
              env_batch=args.env_batch,
              writer=writer)

agent = SAC(env_batch   = args.env_batch,
            max_step    = args.max_step,
            writer      = writer,
            resume      = args.resume,
            output_path = args.output)

print('observation_space', env.observation_space)
print('action_space',      env.action_space)

# Validation 
def validate(step):
    agent.eval()
    all_dists   = []
    blank_dists = []

    for ep in range(args.validate_episodes):
        obs = env.reset(test=True, episode=ep)
        blank_dists.append(np.mean(env.get_dist()))

        for _ in range(args.max_step):
            action = agent.select_action(obs, evaluate=True)
            obs, _, _, _ = env.step(action)

        all_dists.append(np.mean(env.get_dist()))

    mean_blank  = np.mean(blank_dists)
    mean_dist   = np.mean(all_dists)
    improvement = mean_blank - mean_dist

    writer.add_scalar('validate/mean_dist',   mean_dist,   step)
    writer.add_scalar('validate/blank_dist',  mean_blank,  step)
    writer.add_scalar('validate/improvement', improvement, step)

    print(f'[validate] step={step:07d}  '
          f'blank={mean_blank:.4f}  '
          f'painted={mean_dist:.4f}  '
          f'improvement={improvement:+.4f}  '
          f'dist={mean_dist:.4f}  '
          f'{"PAINTING" if improvement > 0.01 else "NOT LEARNING"}')

    agent.save_model(args.output)
    agent.train()
    # no env reset here — caller resets the training env cleanly after


# Training loop
def train():
    obs              = env.reset(test=False)
    step             = args.resume_step
    episode          = step // args.max_step
    session_episodes = 0
    ep_steps         = 0
    interval_t0      = time.time()
    ep_rewards  = []

    a_loss, c_loss, alpha = None, None, 1.0

    while step < args.train_steps:
        action = agent.select_action(obs)

        next_obs, reward, done, _ = env.step(action)
        ep_steps += 1
        ep_rewards.append(float(np.mean(reward)))

        # done flag: only true at episode end
        d = np.ones(args.env_batch, dtype='float32') if ep_steps >= args.max_step \
            else np.zeros(args.env_batch, dtype='float32')
        agent.observe(obs, action, reward, next_obs, d)

        obs   = next_obs
        step += 1  # Increment by 1 loop to mathematically match DDPG's timeline

        if args.debug and step % 10 == 0:
            mean_dist = np.mean(env.get_dist())
            a_str = (f'a_loss={a_loss:.4f}  c_loss={c_loss:.4f}  alpha={alpha:.4f}'
                     if a_loss is not None else 'warming up')
            print(f'  step={step:07d}  ep={episode:04d}  buffer={agent.memory.size()}  '
                  f'reward={np.mean(ep_rewards):.4f}  dist={mean_dist:.4f}  {a_str}')

        # episode end
        if ep_steps >= args.max_step:
            session_episodes += 1
            
            # Warm up for 5 episodes (200 loops) to perfectly fill the 16k replay buffer.
            # Otherwise, the network overfits to blank canvases and instantly destroys its weights.
            if session_episodes > 5:
                if step < 10000 * args.max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * args.max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)

                for _ in range(args.episode_train_times):
                    result = agent.update(lr=lr)
                    if result[0] is not None:
                        a_loss, c_loss, alpha = result

            if args.debug and episode % 10 == 0:
                elapsed     = time.time() - interval_t0
                interval_t0 = time.time()
                print(f'episode={episode:05d}  step={step:07d}  '
                      f'mean_reward={np.mean(ep_rewards):.4f}  '
                      f'alpha={alpha:.4f}  elapsed={elapsed:.1f}s')

            ep_steps   = 0
            episode   += 1
            ep_rewards = []
            obs        = env.reset(test=False)

        # validation
        if ep_steps == 0 and episode > 0 and episode % args.validate_interval == 0:
            validate(step)
            obs = env.reset(test=False)


if __name__ == '__main__':
    train()