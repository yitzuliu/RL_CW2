import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time
from datetime import datetime
from tqdm import tqdm


class TrainingLogger:
    
    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        
       
        self.training_rewards = []
        self.evaluation_rewards = []
        self.evaluation_steps = []
        self.epsilon_values = []
        self.episode_lengths = []
        self.frame_counts = []
        
        
        self.start_time = time.time()
        self.episode_start_time = None
        
       
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
        self.init_log_files()
    
    def init_log_files(self):
        
       
        with open(self.metrics_file, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'training_rewards': [],
                'evaluation_rewards': [],
                'evaluation_steps': [],
                'epsilon_values': [],
                'episode_lengths': [],
                'training_fps': []
            }, f, indent=2)
    
    def log_episode(self, episode, reward, steps, epsilon, loss):
     
        self.training_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.epsilon_values.append(epsilon)
        
       
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        fps = steps / episode_time if episode_time > 0 else 0
        
       
        if episode % self.config.logging_frequency == 0:
            avg_reward = np.mean(self.training_rewards[-self.config.logging_frequency:])
            avg_length = np.mean(self.episode_lengths[-self.config.logging_frequency:])
            
            print(f"Episode {episode}/{self.config.total_episodes} | "
                  f"Reward: {reward:.1f} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Steps: {steps} | "
                  f"Epsilon: {epsilon:.4f} | "
                  f"Loss: {loss:.6f} | "
                  f"FPS: {fps:.1f}")
            
            
            self.update_metrics_file()
    
    def log_evaluation(self, episode, mean_reward, std_reward, steps):
        
        self.evaluation_rewards.append(mean_reward)
        self.evaluation_steps.append(steps)
        
        print(f"\nEvaluation at episode {episode} | "
              f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}\n")
        
       
        self.update_metrics_file()
    
    def update_metrics_file(self):
       
        metrics = {
            'config': self.config.__dict__,
            'training_rewards': self.training_rewards,
            'evaluation_rewards': self.evaluation_rewards,
            'evaluation_steps': self.evaluation_steps,
            'epsilon_values': self.epsilon_values,
            'episode_lengths': self.episode_lengths,
            'elapsed_time': time.time() - self.start_time
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 8))
        
       
        plt.subplot(2, 1, 1)
        plt.plot(self.training_rewards, alpha=0.5, color='blue', label='Episode Reward')
        
       
        if len(self.training_rewards) >= 100:
            moving_avg = np.convolve(self.training_rewards, 
                                    np.ones(100)/100, 
                                    mode='valid')
            plt.plot(range(99, len(self.training_rewards)), 
                    moving_avg, 
                    color='blue', 
                    label='Moving Avg (100 ep)')
        
        
        if self.evaluation_rewards:
            eval_episodes = [i * self.config.evaluation_frequency 
                            for i in range(len(self.evaluation_rewards))]
            plt.plot(eval_episodes, 
                    self.evaluation_rewards, 
                    'ro-', 
                    label='Evaluation Reward')
        
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
       
        plt.subplot(2, 1, 2)
        plt.plot(self.epsilon_values, color='green', label='Exploration Rate (ε)')
        plt.title('Exploration Rate Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def start_episode(self):
        """Record the start time of an episode"""
        self.episode_start_time = time.time()
    
    def save_training_data(self):
        
        np.save(os.path.join(self.log_dir, 'training_rewards.npy'), self.training_rewards)
        np.save(os.path.join(self.log_dir, 'evaluation_rewards.npy'), self.evaluation_rewards)
        np.save(os.path.join(self.log_dir, 'episode_lengths.npy'), self.episode_lengths)
        np.save(os.path.join(self.log_dir, 'epsilon_values.npy'), self.epsilon_values)


def set_random_seeds(seed):
   
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_agent(env, agent, frame_processor, num_episodes):
    
    rewards = []
    total_steps = 0
    
   
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        state = frame_processor.initialize(obs)
        
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            
            obs, reward, done, truncated, _ = env.step(action)
            
        
            state = frame_processor.update(obs)
            
           
            episode_reward += reward
            steps += 1
            
        total_steps += steps
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards), total_steps


def create_video_recorder(env, video_dir, episode_trigger=None):
    
    from gymnasium.wrappers import RecordVideo
    
    
    os.makedirs(video_dir, exist_ok=True)
    
   
    if episode_trigger is None:
        episode_trigger = lambda episode_id: episode_id % 100 == 0
        
   
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=episode_trigger,
        name_prefix="rl-video"
    )
    
    return env