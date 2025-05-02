
import os
import argparse
import torch

from config import create_config
from trainer import DQNTrainer
from utils import set_random_seeds


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='Train a Dueling DQN agent on Atari games'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint file to resume training'
    )
    
    parser.add_argument(
        '--render', 
        action='store_true',
        help='Enable rendering of the environment'
    )
    
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    
    config = create_config()
    
    
    if args.render:
        config.render_mode = 'human'
    
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available with {device_count} device(s)")
        print(f"Using device: {device_name}")
        
        
        if config.use_mixed_precision:
            if torch.cuda.is_bf16_supported():
                print("Mixed precision training enabled with BF16 support")
            elif torch.cuda.is_fp16_supported():
                print("Mixed precision training enabled with FP16 support")
            else:
                print("Warning: Mixed precision requested but may not be fully supported on this GPU")
                config.use_mixed_precision = False
    else:
        print("CUDA not available, using CPU")
        
        config.use_mixed_precision = False
    
    
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    
    set_random_seeds(config.random_seed)
    
    
    trainer = DQNTrainer(config)
    
    
    trainer.train(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()