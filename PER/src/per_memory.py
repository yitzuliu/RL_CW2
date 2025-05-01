import numpy as np
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.sumtree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PERMemory:
    def __init__(self, memory_capacity=config.MEMORY_CAPACITY, 
                 alpha=config.ALPHA, 
                 beta_start=config.BETA_START, 
                 beta_frames=config.BETA_FRAMES,
                 epsilon=config.EPSILON_PER):
        self.sumtree = SumTree(memory_capacity)
        self.memory_capacity = memory_capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.beta = beta_start
        self.frame_count = 0
        self.max_priority = 1.0
    
    def update_beta(self, frame_idx):
        self.frame_count = frame_idx
        progress = min(1.0, frame_idx / self.beta_frames)
        adjusted_progress = progress ** config.BETA_EXPONENT
        self.beta = min(1.0, self.beta_start + adjusted_progress * (1.0 - self.beta_start))
    
    def _calculate_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        transition = Transition(state, action, reward, next_state, done)
        if td_error is None:
            priority = self.max_priority
        else:
            priority = self._calculate_priority(td_error)
            self.max_priority = max(self.max_priority, priority)
        self.sumtree.add(priority, transition)
    
    def sample(self, batch_size):
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch_weights = np.zeros(batch_size, dtype=np.float32)
        batch_transitions = []
        segment_size = self.sumtree.total_priority() / batch_size
        current_beta = self.beta
        min_prob = np.min(self.sumtree.get_all_priorities()) / self.sumtree.total_priority()
        max_weight = (min_prob * self.sumtree.experience_count) ** (-current_beta)
        for i in range(batch_size):
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            value = np.random.uniform(segment_start, segment_end)
            idx, priority, transition = self.sumtree.get_experience_by_priority(value)
            batch_indices[i] = idx
            sample_prob = priority / self.sumtree.total_priority()
            weight = (sample_prob * self.sumtree.experience_count) ** (-current_beta)
            batch_weights[i] = weight / max_weight
            batch_transitions.append(transition)
        return batch_indices, batch_weights, batch_transitions
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = self._calculate_priority(error)
            self.max_priority = max(self.max_priority, priority)
            self.sumtree.update_priority(idx, priority)
    
    def __len__(self):
        return self.sumtree.experience_count

