import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.device_utils import get_device

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_space_size, 
                 use_one_conv=config.USE_ONE_CONV_LAYER,
                 use_two_conv=config.USE_TWO_CONV_LAYERS,
                 use_three_conv=config.USE_THREE_CONV_LAYERS):
        super(QNetwork, self).__init__()
        self.use_one_conv = use_one_conv
        self.use_two_conv = use_two_conv
        self.use_three_conv = use_three_conv
        in_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels, config.CONV1_CHANNELS, 
                               kernel_size=config.CONV1_KERNEL_SIZE, 
                               stride=config.CONV1_STRIDE)
        if use_two_conv or use_three_conv:
            self.conv2 = nn.Conv2d(config.CONV1_CHANNELS, config.CONV2_CHANNELS, 
                                   kernel_size=config.CONV2_KERNEL_SIZE, 
                                   stride=config.CONV2_STRIDE)
        if use_three_conv:
            self.conv3 = nn.Conv2d(config.CONV2_CHANNELS, config.CONV3_CHANNELS, 
                                   kernel_size=config.CONV3_KERNEL_SIZE, 
                                   stride=config.CONV3_STRIDE)
        if use_three_conv:
            self.feature_size = self._get_conv_output_size(input_shape)
        elif use_two_conv:
            self.feature_size = self._get_conv_output_size(input_shape, use_three_conv=False)
        else:
            self.feature_size = self._get_conv_output_size(input_shape, use_two_conv=False, use_three_conv=False)
        self.fc1 = nn.Linear(self.feature_size, config.FC_SIZE)
        self.fc2 = nn.Linear(config.FC_SIZE, action_space_size)
        self._initialize_weights()
        self.device = get_device()
        self.to(self.device)
    
    def _get_conv_output_size(self, input_shape, use_two_conv=True, use_three_conv=True):
        x = torch.zeros(1, *input_shape)
        x = F.relu(self.conv1(x))
        if use_two_conv:
            x = F.relu(self.conv2(x))
        if use_three_conv:
            x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        if x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.dim() == 4 and x.shape[1] not in [1, 4]:
            x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        if self.use_two_conv or self.use_three_conv:
            x = F.relu(self.conv2(x))
        if self.use_three_conv:
            x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
