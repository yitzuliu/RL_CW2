import gymnasium as gym
import numpy as np
from collections import deque


class FrameSkipWrapper(gym.Wrapper):
  
    
    def __init__(self, env, skip=4, max_pool=False):
       
        super(FrameSkipWrapper, self).__init__(env)
        self.skip = skip
        self.max_pool = max_pool
        self.obs_buffer = deque(maxlen=2)  
        
    def step(self, action):
       
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        
        for i in range(self.skip):
            observation, reward, term, trunc, step_info = self.env.step(action)
            
            
            if self.max_pool:
                self.obs_buffer.append(observation)
                
            total_reward += reward
            terminated = term
            truncated = trunc
            info = step_info  
            
            
            if terminated or truncated:
                break
        
        if self.max_pool and len(self.obs_buffer) > 1:
            observation = np.max(np.stack(self.obs_buffer), axis=0)
            
        return observation, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        
        observation, info = self.env.reset(**kwargs)
        
        if self.max_pool:
            self.obs_buffer.clear()
            self.obs_buffer.append(observation)
            
        return observation, info


class EpisodicLifeWrapper(gym.Wrapper):
   
    
    def __init__(self, env):
       
        super(EpisodicLifeWrapper, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        """
        Check for life loss and treat as terminal state when it occurs.
        
        Args:
            action: The action to take
            
        Returns:
            Modified (observation, reward, terminated, truncated, info)
        """