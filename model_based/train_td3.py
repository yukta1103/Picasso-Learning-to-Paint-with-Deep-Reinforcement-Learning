print('>>> Starting TD3 Training...', flush=True)

import os
import sys
import cv2
import random
import time
import torch
import pickle
import argparse
import numpy as np
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
from DRL.td3 import TD3
from DRL.multi import fastenv

exp = os.path.basename(os.path.abspath('.'))
log_dir = os.path.join('..', 'train_log', f'{exp}_td3')
os.makedirs(log_dir, exist_ok=True)
writer = TensorBoard(log_dir)
os.makedirs('./model_td3', exist_ok=True)

CHECKPOINT_INTERVAL = 5000 


def save_checkpoint(agent, step, episode, output):
    agent.save_model(output)
    state = {'step': step, 'episode': episode}
    with open(os.path.join(output, 'train_state.pkl'), 'wb') as f:
        pickle.dump(state, f)
    with open(os.path.join(output, 'replay_buffer.pkl'), 'wb') as f:
        pickle.dump(agent.memory, f)
    print(f'>>> Checkpoint saved at step {step}', flush=True)


def load_checkpoint(output):
    state_path = os.path.join(output, 'train_state.pkl')
    if not os.path.exists(state_path):
        return 0, 0
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    print(f'>>> Resuming from step {state["step"]}, episode {state["episode"]}', flush=True)
    return state['step'], state['episode']


def load_replay_buffer(agent, output):
    buf_path = os.path.join(output, 'replay_buffer.pkl')
    if not os.path.exists(buf_path):
        print('>>> No replay buffer found, starting fresh.', flush=True)
        return
    with open(buf_path, 'rb') as f:
        agent.memory = pickle.load(f)
    print(f'>>> Replay buffer loaded successfully.', flush=True)


def train(agent, env, evaluate, args):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    output = args.output
    time_stamp = time.time()
    noise_factor = args.noise_factor

    step, episode = load_checkpoint(output)
    episode_steps = 0
    tot_reward = 0.
    observation = None

    if step > 0:
        load_replay_buffer(agent, output)

    while step <= train_times:
        step += 1
        episode_steps += 1

        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)

        if episode_steps >= max_step and max_step:
            if step > args.warmup:
                if episode > 0 and validate_interval > 0 and \
                        episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    mean_reward = np.mean(reward)
                    mean_dist = np.mean(dist)
                    if debug:
                        prRed(
                            'Step_{:07d}: mean_reward:{:.3f} '
                            'mean_dist:{:.3f} var_dist:{:.3f}'.format(
                                step - 1, mean_reward,
                                mean_dist, np.var(dist)))
                    if writer:
                        writer.add_scalar('validate/mean_reward', mean_reward, step)
                        writer.add_scalar('validate/mean_dist', mean_dist, step)
                        writer.add_scalar('validate/var_dist', np.var(dist), step)
                    save_checkpoint(agent, step, episode, output)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.

            if step > args.warmup:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                if writer:
                    writer.add_scalar('train/critic_lr', lr[0], step)
                    writer.add_scalar('train/actor_lr', lr[1], step)
                    writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                    writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)

                if step % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(agent, step, episode, output)

            if debug:
                prBlack('#{}: steps:{} interval_time:{:.2f} '
                        'train_time:{:.2f}'.format(
                            episode, step, train_time_interval,
                            time.time() - time_stamp))
            time_stamp = time.time()
            observation = None
            episode_steps = 0
            episode += 1

    save_checkpoint(agent, step, episode, output)
    print('>>> Training complete.', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint — TD3')

    parser.add_argument('--warmup', default=400, type=int)
    parser.add_argument('--discount', default=0.95**5, type=float)
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--rmsize', default=400, type=int)
    parser.add_argument('--env_batch', default=96, type=int)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--max_step', default=40, type=int)
    parser.add_argument('--noise_factor', default=0.1, type=float)  # was 0, needs exploration
    parser.add_argument('--validate_interval', default=50, type=int)
    parser.add_argument('--validate_episodes', default=5, type=int)
    parser.add_argument('--train_times', default=100000, type=int)  # run in chunks
    parser.add_argument('--episode_train_times', default=10, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--output', default='./model_td3', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    args = parser.parse_args()
    if not args.cuda:
        os.environ["USE_CUDA"] = "0"

    import utils.util
    utils.util.device = utils.util.get_device()
    from utils.util import device, prRed, prBlack, get_output_folder
    
    if args.resume is not None:
        args.output = args.resume
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = get_output_folder(args.output, "Paint-TD3")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print(f'Using Device: {device}', flush=True)
    print(f'Output folder: {args.output}', flush=True)
    print('Initializing environment and agent...', flush=True)

    fenv = fastenv(args.max_step, args.env_batch, writer)
    agent = TD3(
        batch_size=args.batch_size,
        env_batch=args.env_batch,
        max_step=args.max_step,
        tau=args.tau,
        discount=args.discount,
        rmsize=args.rmsize,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        writer=writer,
        resume=args.resume,
        output_path=args.output,
    )
    evaluate = Evaluator(args, writer)

    train(agent, fenv, evaluate, args)