import random
import numpy as np
import torch
from collections import deque


class ExperienceReplayMemory:
    """
    Experience replay memory buffer for deep reinforcement learning.
    
    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training.
    """
    def __init__(self, capacity, device="cuda"):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            device: Device to send tensors to (cuda or cpu)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device
        
    def store_transition(self, state, action, reward, next_state, done):
        """
        Add a new transition to the memory buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_batch(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        transitions = random.sample(self.memory, batch_size)
        
      
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def can_sample(self, batch_size):
        """
        Check if enough transitions are stored to sample a batch.
        
        Args:
            batch_size: Requested batch size
            
        Returns:
            True if there are enough samples, False otherwise
        """
        return len(self.memory) >= batch_size
    
    def __len__(self):
        """Return the current size of the memory buffer."""
        return len(self.memory)
        
    def is_full(self):
        """Check if the memory buffer is at capacity."""
        return len(self.memory) == self.capacity
        
    def clear(self):
        """Clear all transitions from memory."""
        self.memory.clear()