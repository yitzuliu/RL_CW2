import os
import time
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

from preprocessing import AtariFrameProcessor
from memory import ExperienceReplayMemory
from agent import DuelingDQNAgent
from utils import TrainingLogger, evaluate_agent, create_video_recorder


class DQNTrainer:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Starting training with configuration:")
        print(config)
        print(f"Using device: {self.device}")
        
        self.env = self._make_atari_environment(config.env_name, config.render_mode)
        
        if config.enable_evaluation:
            self.eval_env = self._make_atari_environment(config.env_name)
        else:
            self.eval_env = None
            
       
        if config.render_mode is not None:
            video_dir = os.path.join(config.log_dir, "videos")
            self.env = create_video_recorder(
                self.env, 
                video_dir, 
                episode_trigger=lambda ep: ep % 100 == 0
            )
        
        
        self.frame_processor = AtariFrameProcessor(
            frame_history=config.frame_stack,
            frame_dimensions=config.frame_size
        )
        
        if config.enable_evaluation:
            self.eval_frame_processor = AtariFrameProcessor(
                frame_history=config.frame_stack,
                frame_dimensions=config.frame_size
            )
        
        
        self.memory = ExperienceReplayMemory(
            capacity=config.memory_capacity,
            device=self.device
        )
        
        
        input_shape = (config.frame_stack, config.frame_size[0], config.frame_size[1])
        action_count = self.env.action_space.n
        
        self.agent = DuelingDQNAgent(
            config=config,
            state_shape=input_shape,
            action_count=action_count,
            device=self.device
        )
        
    
        self.logger = TrainingLogger(config)
        
    
        self.current_episode = 0
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
    def _make_atari_environment(self, env_name, render_mode=None):
        from wrappers import FrameSkipWrapper, EpisodicLifeWrapper, FireResetWrapper
        
       
        env = gym.make(env_name, render_mode=render_mode)
        
        if 'ALE/' in env_name:
           
            env = FrameSkipWrapper(env, skip=4, max_pool=True)
            
        
            env = EpisodicLifeWrapper(env)
            

            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetWrapper(env)
        
        return env
    
    def train(self, checkpoint_path=None):
    
 
        if checkpoint_path:
            episode = self.agent.load_checkpoint(checkpoint_path)
            if episode:
                self.current_episode = episode
                print(f"Resuming from episode {self.current_episode}")
        
        try:

            for episode in range(self.current_episode + 1, self.config.total_episodes + 1):
                self.current_episode = episode
                
          
                self._run_episode(episode)
            
                if self.config.enable_evaluation and episode % self.config.evaluation_frequency == 0:
                    self._evaluate_agent(episode)
                

                if episode % self.config.checkpoint_frequency == 0:
                    checkpoint_path = os.path.join(
                        self.config.model_dir, 
                        f"checkpoint_episode_{episode}.pt"
                    )
                    self.agent.save_checkpoint(checkpoint_path, episode)
                    print(f"Saved checkpoint at episode {episode}")
                
                if episode % 10 == 0:
                    self.logger.plot_training_progress()
            
         
            self.agent.save_checkpoint(
                os.path.join(self.config.model_dir, "final_model.pt"),
                self.config.total_episodes
            )
            
            print("Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
          
            self.env.close()
            if self.eval_env:
                self.eval_env.close()
            
           
            self.logger.save_training_data()
            self.logger.plot_training_progress()
    
    def _run_episode(self, episode):

       
        obs, _ = self.env.reset()
        state = self.frame_processor.initialize(obs)
        
       
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        done = False
        truncated = False
        
       
        self.logger.start_episode()
        
      
        while not (done or truncated):
            action = self.agent.select_action(state)
            
       
            obs, reward, done, truncated, info = self.env.step(action)
            
          
            next_state = self.frame_processor.update(obs)
            
            
            self.memory.store_transition(state, action, reward, next_state, done or truncated)
            
           
            state = next_state
            
          
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            
            if (self.total_steps % self.config.train_frequency == 0 and 
                self.memory.can_sample(self.config.batch_size)):
                
                
                experiences = self.memory.sample_batch(self.config.batch_size)
                loss = self.agent.train(experiences)
                episode_loss.append(loss)
        
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
     
        epsilon = self.agent._get_current_epsilon()
        
        
        self.logger.log_episode(
            episode=episode,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=epsilon,
            loss=avg_loss
        )
    
    def _evaluate_agent(self, episode):
        mean_reward, std_reward, eval_steps = evaluate_agent(
            env=self.eval_env,
            agent=self.agent,
            frame_processor=self.eval_frame_processor,
            num_episodes=self.config.evaluation_episodes
        )
        
       
        self.logger.log_evaluation(
            episode=episode,
            mean_reward=mean_reward,
            std_reward=std_reward,
            steps=self.total_steps
        )
        
      
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            self.agent.save_checkpoint(
                os.path.join(self.config.model_dir, "best_model.pt"),
                episode
            )
            print(f"New best model saved with reward {mean_reward:.2f}")