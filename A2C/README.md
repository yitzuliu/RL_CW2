# Reinforcement-Learning-Coursework-2

# A2C Atari Agent – MsPacman

This repository contains a simple implementation of an -- Advantage Actor-Critic (A2C) -- agent for the Atari game Ms. Pacman, using `gymnasium`, `ale-py`, and `PyTorch`. The code includes an environment factory, a convolutional neural network with shared policy/value heads, and a training loop with debugging outputs.

## Environment

This code was developed and tested in -- Kaggle Colab -- (Kaggle's hosted Jupyter notebook environment, similar to Google Colab).  
**Important:**  
Because of this, the dependencies and library versions are aligned with the Kaggle/Colab environment. Running this code outside of Kaggle/Colab (e.g., on a local machine or a different cloud setup) [might result in version mismatches, missing packages, or crashes.]

Key dependencies:
- `gymnasium`
- `ale-py`
- `torch`
- `numpy`

## How to Run

To run the code:
1. Copy the entire Python script into a Kaggle Notebook or Colab Notebook.
2. Make sure to install the required packages (most are pre-installed in Kaggle):
   ```bash
   pip install gymnasium ale-py torch numpy
