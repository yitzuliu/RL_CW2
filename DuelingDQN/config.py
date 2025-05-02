import os
import argparse
import yaml
from datetime import datetime


class DQNConfiguration:
    
    def __init__(self, config_file=None):
      
        self.env_name = 'ALE/MsPacman-v5'
        self.render_mode = None
        
        self.frame_stack = 4
        self.frame_size = (84, 84)
        
     
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.discount_factor = 0.99
        self.start_exploration_rate = 1.0
        self.end_exploration_rate = 0.1
        self.exploration_decay_steps = 500000
        self.target_sync_frequency = 10000
        self.train_frequency = 4
        self.memory_capacity = 100000
        self.min_memory_size = 10000
        self.use_double_q = True
        self.use_mixed_precision = True
        
        
        self.total_episodes = 1000
        self.max_steps_per_episode = 10000
        self.checkpoint_frequency = 100
        self.logging_frequency = 1000
        self.enable_evaluation = True
        self.evaluation_frequency = 100
        self.evaluation_episodes = 10
        
       
        self.random_seed = 42
        
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"dueling_dqn_mspacman_{timestamp}"
        self.model_dir = os.path.join("models", self.run_name)
        self.log_dir = os.path.join("logs", self.run_name)
        
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
            
       
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, file_path):
        config_dict = {key: value for key, value in self.__dict__.items()}
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self):
        config_str = "=== Training Configuration ===\n"
        for key, value in self.__dict__.items():
            config_str += f"{key}: {value}\n"
        return config_str


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dueling DQN for Atari games')
    
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--env', type=str, help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--memory-size', type=int, help='Replay memory capacity')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--eval', action='store_true', help='Enable evaluation')
    parser.add_argument('--disable-eval', action='store_true', help='Disable evaluation')
    
    return parser.parse_args()


def create_config():
    args = parse_arguments()
    config = DQNConfiguration(config_file=args.config)
    
    if args.env:
        config.env_name = args.env
    if args.episodes:
        config.total_episodes = args.episodes
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.memory_size:
        config.memory_capacity = args.memory_size
    if args.seed:
        config.random_seed = args.seed
    if args.eval:
        config.enable_evaluation = True
    if args.disable_eval:
        config.enable_evaluation = False
    
    return config


if __name__ == "__main__":
    config = create_config()
    config.save_to_file("default_config.yaml")
    print(config)