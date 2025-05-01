# Prioritized Experience Replay (PER) for Atari Games

Training a Deep Q-Network (DQN) to play Atari games using Prioritized Experience Replay (PER) technique.

![Atari Game Example](https://gymnasium.farama.org/_images/ms_pacman.gif)

## 📝 Project Overview

This project implements a Deep Q-Network (DQN) with Prioritized Experience Replay (PER) to play Atari games. PER is an enhanced experience replay mechanism that prioritizes sampling of high-value experiences based on their importance (measured by TD-error). This approach significantly improves DQN's learning efficiency and performance.
## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Gymnasium (newer version of OpenAI Gym)
- NumPy, Matplotlib, OpenCV
- Other dependencies listed in requirements.txt

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/atari-per-dqn.git
cd atari-per-dqn
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## 📊 Project Structure

```
.
├── config.py                    # Configuration file with all hyperparameters
├── train.py                     # Training script to start the training process
├── src/
│   ├── dqn_agent.py             # DQN agent implementation
│   ├── per_memory.py            # Prioritized Experience Replay memory
│   ├── sumtree.py               # SumTree data structure implementation
│   ├── q_network.py             # Q-Network neural network architecture
│   ├── env_wrappers.py          # Atari environment wrappers
│   ├── device_utils.py          # Device detection and optimization utilities
│   ├── logger.py                # Training logging tools
│   └── visualization.py         # Training metrics visualization tools
├── results/                      # Directory for storing training results
│   ├── data/                    # Training data
│   ├── logs/                    # Training logs
│   ├── models/                  # Saved models
│   └── plots/                   # Generated plots
└── CHECKLIST.md                 # Implementation plan and pseudocode
```

## 🚀 Usage

### Training a Model

To train a new model from scratch:

```bash
python train.py
```
#### Example
```bash
python visualize_agent.py --exp_id exp_20250430_014335 --checkpoint checkpoint_1000.pt --speed 0.5 --episodes 5 --difficulty 2
```

- `--exp_id`: Specify the experiment ID (e.g., `exp_20250430_014335`).
- `--checkpoint`: Specify the checkpoint file to load (e.g., `checkpoint_1000.pt`).
- `--speed`: Adjust the game speed (e.g., `0.5` for slow motion).
- `--episodes`: Number of episodes to visualize (default: 3).
- `--difficulty`: Set the game difficulty level (0-4).

### Algorithm Description

This project implements the DQN algorithm with Prioritized Experience Replay:

1. **Prioritized Experience Replay (PER)**:
    - Efficiently stores and samples experiences using a SumTree data structure
    - Transitions are assigned priorities based on TD-error: p = (|δ|+ε)^α
    - Importance sampling weights w = (N·P(i))^(-β) correct the bias introduced
    - β gradually increases from 0.4 to 1.0 over time

2. **DQN Architecture**:
    - 3-layer convolutional neural network followed by fully connected layers
    - Dual network architecture (policy and target networks) with PER integration
    - ε-greedy exploration strategy with decaying ε value
    - Target network updates every 20,000 steps

## 🔍 Algorithm Details

### SumTree Data Structure

The implemented SumTree has the following main operations:
- `add`: Add new experience with its priority
- `update_priority`: Update the priority of an experience
- `get_experience_by_priority`: Retrieve experience based on priority value

### Prioritized Experience Replay Mechanism

The PER memory implements:
- Priority calculation based on TD-errors
- Computation and proper application of importance sampling weights
- Linear annealing strategy for β value
- Efficient batch sampling