# Dueling Double DQN for Ms. Pac-Man

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium with Atari environments
- OpenCV, NumPy, Matplotlib, tqdm, PyYAML

## How to run

1. Install dependencies:
```bash
pip install torch gymnasium[atari] gymnasium[accept-rom-license] opencv-python numpy matplotlib tqdm pyyaml
```

2. Start training:
```bash
python main.py
```

3. Evaluate a trained model:
```bash
python run_evaluation.py --model models/dueling_dqn_mspacman/best_model.pt
```

## Hyperparameters

Key hyperparameters:
- Learning rate: 0.00025
- Discount factor (γ): 0.99
- Batch size: 32
- Epsilon decay: 1.0 → 0.1 over 500K frames
- Experience replay: 100K transitions
- Target network update: Every 10K steps

Modify settings through command-line arguments or create a custom YAML configuration file.