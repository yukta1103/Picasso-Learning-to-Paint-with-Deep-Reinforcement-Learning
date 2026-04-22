# Picasso: Learning to Paint with Deep Reinforcement Learning

Course project for CS5180 - Reinforcement Learning and Sequential Decision Making, Northeastern University.

This project reimplements and extends the LearningToPaint framework (Huang et al., ICCV 2019) by replacing the original DDPG agent with TD3 and SAC. The agent learns to reconstruct images by sequentially placing brush strokes on a canvas, treating painting as a continuous control problem.

---

## Results

The agent was trained on CelebA face images for 3 million steps. Both algorithms successfully learn to reconstruct target images through stroke placement. SAC produces visibly sharper and more accurate reconstructions compared to TD3, which exhibits grainier texture artifacts.

---

## How It Works

The environment presents the agent with a target image and a blank canvas. At each step, the agent outputs a 65-dimensional (5 strokes × 13 params each) action representing a brush stroke (Bezier control points, radius, and RGB color). A pretrained neural renderer converts these parameters into actual pixels on the canvas. The reward at each step is the reduction in L2 distance between the current canvas and the target image. Episodes run for a fixed number of steps (default 40).

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
