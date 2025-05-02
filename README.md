# Reinforcement Learning Algorithms for Atari Games

This repository contains implementations of various state-of-the-art reinforcement learning algorithms for playing Atari games, with a focus on Ms. Pac-Man.

## Project Overview

This project implements and compares different reinforcement learning approaches:

- **Prioritized Experience Replay (PER)** - Enhanced version of DQN using prioritized sampling
- **Dueling DQN** - Network architecture that separates state value and advantage estimations
- **Advantage Actor-Critic (A2C)** - Policy gradient method with baseline function
- **Proximal Policy Optimization (PPO)** - Policy optimization with clipped surrogate objective
- **Rainbow DQN** - Enhanced version of DQN with 6 key improvements

## Repository Structure

```
.
├── A2C/                      
│   ├── A2C.ipynb             
│   └── README.md             
├── DuelingDQN/               
│   ├── agent.py              
│   ├── config.py            
│   ├── main.py               
│   ├── memory.py             
│   ├── models.py             
│   └── ...                   
├── PER/                     
│   ├── train.py              
│   ├── src/                  
│   │   ├── dqn_agent.py      
│   │   ├── per_memory.py     
│   │   ├── sumtree.py        
│   │   └── ...              
│   └── README.md             
├── ppo/                      
│   ├── config.py             
│   ├── main.py               
│   ├── ppo_cnn.py
│   ├── chart.py
│   ├── chart_seperate.py  
│   ├── demo.py
│   ├── hex_result/
│   │   ├── actor_loss_record.npy
│   │   ├── critic_loss_record.npy
│   │   ├── epsilon_record.npy
│   │   ├── log_*.txt files
│   │   └── ...
│   └── ...                   
└── RainbowDQN/
    ├── agent.py
    ├── environment.py
    ├── main.py
    ├── memory_replay.py
    ├── model.py
    ├── utils.py
    ├── test.py
    └── requirements.txt
```

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Gymnasium (newer version of OpenAI Gym)
- ale-py (Arcade Learning Environment)
- NumPy, Matplotlib, OpenCV
