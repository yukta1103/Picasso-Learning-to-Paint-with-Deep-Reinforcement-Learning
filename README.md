# Picasso: Learning to Paint with Deep Reinforcement Learning

Course project for CS5180 - Reinforcement Learning and Sequential Decision Making, Northeastern University.

This project reimplements and extends the LearningToPaint framework (Huang et al., ICCV 2019) by replacing the original DDPG agent with TD3 and SAC. The agent learns to reconstruct images by sequentially placing brush strokes on a canvas, treating painting as a continuous control problem.

---

## Results

The agent was trained on CelebA face images for 3 million steps. Both algorithms successfully learn to reconstruct target images through stroke placement. SAC produces visibly sharper and more accurate reconstructions compared to TD3, which exhibits grainier texture artifacts.

---

## How It Works

The environment presents the agent with a target image and a blank canvas. At each step, the agent outputs a 13-dimensional action representing a brush stroke (Bezier control points, radius, and RGBA color). A pretrained neural renderer converts these parameters into actual pixels on the canvas. The reward at each step is the reduction in L2 distance between the current canvas and the target image. Episodes run for a fixed number of steps (default 40).

The neural renderer (renderer.pkl) is frozen during RL training. It was pretrained separately to map stroke parameter vectors to rasterized stroke images.

---

## Repository Structure

```
Picasso-Learning-to-Paint-with-Deep-Reinforcement-Learning/
|
|-- model_based/
|   |
|   |-- DRL/
|   |   |-- actor.py            # Actor network for TD3 (deterministic, tanh output)
|   |   |-- critic.py           # Twin Q-networks for TD3
|   |   |-- actor_sac.py        # Stochastic actor for SAC (Gaussian policy, reparameterization)
|   |   |-- critic_sac.py       # Twin soft Q-networks for SAC
|   |   |-- td3.py              # TD3 agent: twin critics, delayed policy updates, target smoothing
|   |   |-- sac.py              # SAC agent: entropy-regularized actor-critic, auto alpha tuning
|   |   |-- evaluator.py        # Periodic evaluation during training, saves canvas outputs
|   |   |-- rpm.py              # Replay memory buffer used by both TD3 and SAC
|   |   |-- multi.py            # Utilities for batched environment stepping
|   |   |-- wgan.py             # WGAN discriminator used as an auxiliary perceptual reward
|   |
|   |-- Renderer/
|   |   |-- __init__.py
|   |   |-- model.py            # Neural renderer architecture (stroke parameter -> raster image)
|   |   |-- stroke_gen.py       # Procedural Bezier stroke generator (used to train renderer)
|   |
|   |-- utils/
|   |   |-- tensorboard.py      # TensorBoard logging wrapper
|   |   |-- util.py             # L2 loss, image normalization, canvas utilities
|   |
|   |-- env.py                  # Painting environment: step(), reset(), reward computation
|   |-- train_td3.py            # TD3 training entry point
|   |-- train_sac.py            # SAC training entry point
|   |-- test_sac.py             # SAC evaluation: loads checkpoint, renders output image
|   |-- test.py                 # General evaluation script (works with either agent)
|
|-- renderer.pkl                # Pretrained neural renderer weights (required at runtime)
|-- requirements.txt
|-- .gitignore
```

---

## File Descriptions

**env.py**
The core RL environment. Wraps the neural renderer into a Gym-style interface. Maintains the canvas state, loads target images from CelebA, computes per-step L2 improvement rewards, and handles episode termination. Both training scripts import this directly.

**DRL/td3.py**
Implements Twin Delayed DDPG (TD3). Uses two independent Q-networks to reduce overestimation bias, delays actor updates to every 2nd critic step, and adds clipped Gaussian noise to target actions during critic updates (target policy smoothing).

**DRL/sac.py**
Implements Soft Actor-Critic (SAC). Maximizes a maximum-entropy objective that encourages both high reward and high action entropy. Uses the reparameterization trick for the stochastic actor and automatically tunes the temperature parameter alpha via a target entropy constraint.

**DRL/actor.py / DRL/critic.py**
Network definitions for TD3. The actor is a deterministic MLP that outputs a 13-dim action squashed through tanh. The critic consists of two independent MLP heads, each taking a (state, action) pair and outputting a scalar Q-value.

**DRL/actor_sac.py / DRL/critic_sac.py**
Network definitions for SAC. The actor outputs the mean and log-std of a Gaussian distribution over actions; samples are squashed through tanh. The critics are soft Q-networks whose targets incorporate the entropy bonus from the current policy.

**DRL/rpm.py**
Replay memory buffer. Stores (state, action, reward, next_state, done) transitions and supports random sampling for off-policy updates. Used by both TD3 and SAC.

**DRL/evaluator.py**
Runs periodic evaluation episodes during training without exploration noise. Saves rendered canvas images to disk so you can visually track reconstruction quality over the course of training.

**DRL/multi.py**
Handles batched environment interaction. Allows the training loop to step multiple (canvas, target) pairs to improve sample throughput.

**DRL/wgan.py**
A Wasserstein GAN discriminator used as an auxiliary reward signal. Provides a perceptual quality signal on top of the base L2 reward, encouraging the agent to produce more realistic-looking strokes.

**Renderer/model.py**
Architecture of the pretrained neural renderer. A fully connected network that maps a stroke parameter vector to a rasterized stroke image patch. Loaded from renderer.pkl at the start of training and kept frozen.

**Renderer/stroke_gen.py**
Procedural Bezier stroke generator used during renderer pretraining to create synthetic (parameter, raster) training pairs. Not used during RL training.

**utils/util.py**
Shared utilities: L2 distance computation between canvas and target, image tensor normalization, and numpy/PyTorch conversion helpers.

**utils/tensorboard.py**
Thin wrapper around SummaryWriter for logging training metrics (reward, actor loss, critic loss, alpha) to TensorBoard.

**train_td3.py / train_sac.py**
Training entry points. Each script initializes the environment and agent, warms up the replay buffer with random transitions, then runs the standard off-policy training loop with periodic evaluation and checkpoint saving.

**test_sac.py / test.py**
Evaluation scripts. Load a saved checkpoint and run the agent on a test image, saving the final reconstructed canvas for visual comparison against the target.

---

## Setup

Requirements: Python 3.8+, PyTorch, OpenCV, Pillow, NumPy.

Install dependencies:

```bash
pip install -r requirements.txt
```

The pretrained renderer weights (renderer.pkl) are included in the root of the repository. Both training scripts expect it at that path by default.

---

## Training

Train with TD3:

```bash
cd model_based
python train_td3.py
```

Train with SAC:

```bash
cd model_based
python train_sac.py
```

Both scripts train on CelebA. Set the dataset path by editing the img_path or data_root variable near the top of the training script before running.

Checkpoints are saved periodically to ./checkpoints/. Training was run for 3 million environment steps.

---

## Evaluation

Evaluate a trained SAC model:

```bash
cd model_based
python test_sac.py
```

For a general evaluation compatible with either agent:

```bash
cd model_based
python test.py
```

Make sure the checkpoint path inside the evaluation script points to your saved model weights before running.

---

## Algorithm Comparison

| | TD3 | SAC |
|---|---|---|
| Policy type | Deterministic | Stochastic (Gaussian) |
| Exploration | Gaussian noise on actions | Entropy maximization |
| Critic | Twin Q-networks | Twin soft Q-networks |
| Actor update | Delayed (every 2 steps) | Every step |
| Temperature | N/A | Auto-tuned alpha |
| Result quality | Grainy, less accurate | Sharper, more accurate |

SAC outperforms TD3 on this task likely because entropy-driven exploration produces better coverage of the stroke parameter space during training, leading to a more accurate policy at convergence.

---

## Based On

Huang, Z., Heng, W., and Zhou, S. "Learning to Paint with Model-based Deep Reinforcement Learning." ICCV 2019. - https://arxiv.org/pdf/1903.04411