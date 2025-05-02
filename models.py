import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingQNetwork(nn.Module):
  
    
    def __init__(self, input_channels, num_actions):
        
        super(DuelingQNetwork, self).__init__()
        
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        
        self.feature_size = self._calculate_conv_output_size(input_channels)
        
        
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  
        )
    
    def _calculate_conv_output_size(self, input_channels):
        
        
        x = torch.zeros(1, input_channels, 84, 84)
        x = self.feature_extractor(x)
        return int(np.prod(x.size()))
    
    def forward(self, state):
       
       
        features = self.feature_extractor(state)
        
        
        features = features.view(features.size(0), -1)
        
        
        state_value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = state_value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values