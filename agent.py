import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import time

from models import DuelingQNetwork


class DuelingDQNAgent:
  
    
    def __init__(self, config, state_shape, action_count, device):
   
        self.config = config
        self.state_shape = state_shape
        self.action_count = action_count
        self.device = device
        
        self.total_steps = 0
        
        self.policy_network = DuelingQNetwork(state_shape[0], action_count).to(device)
        self.target_network = DuelingQNetwork(state_shape[0], action_count).to(device)
        
        self.sync_target_network()
     
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=config.learning_rate
        )
        
       
        self.training_losses = []
        self.recent_losses = []  
    
    def select_action(self, state, evaluate=False):
       
       
        if evaluate:
            return self._greedy_action(state)
        
        
        epsilon = self._get_current_epsilon()
        
        
        if random.random() < epsilon:
           
            return random.randint(0, self.action_count - 1)
        else:
          
            return self._greedy_action(state)
    
    def _greedy_action(self, state):
        
       
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        
      
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        
        
        return q_values.argmax(dim=1).item()
    
    def _get_current_epsilon(self):
       
        
        decay_progress = min(1.0, self.total_steps / self.config.exploration_decay_steps)
        
        return self.config.end_exploration_rate + \
               (self.config.start_exploration_rate - self.config.end_exploration_rate) * \
               (1 - decay_progress)
    
    def train(self, experiences):
       
        states, actions, rewards, next_states, dones = experiences
        
      
        if self.config.use_mixed_precision and self.device == torch.device("cuda"):
            scaler = torch.cuda.amp.GradScaler()
            
           
            with torch.cuda.amp.autocast():
                
                current_q_values = self.policy_network(states)
                
                
                selected_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                
                with torch.no_grad():
                    if self.config.use_double_q:
                       
                        policy_next_q_values = self.policy_network(next_states)
                        best_actions = policy_next_q_values.argmax(dim=1, keepdim=True)
                        
                        
                        target_next_q_values = self.target_network(next_states)
                        next_q_values = target_next_q_values.gather(1, best_actions).squeeze(1)
                    else:
                       
                        target_next_q_values = self.target_network(next_states)
                        next_q_values = target_next_q_values.max(1)[0]
                    
                 
                    expected_q_values = rewards + \
                                       (1 - dones) * \
                                       self.config.discount_factor * \
                                       next_q_values
                
                loss = F.smooth_l1_loss(selected_q_values, expected_q_values)
            
            
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            
           
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
            
          
            scaler.step(self.optimizer)
            scaler.update()
            
        else:
           
            current_q_values = self.policy_network(states)
            
           
            selected_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            
            with torch.no_grad():
                if self.config.use_double_q:
                    
                    policy_next_q_values = self.policy_network(next_states)
                    best_actions = policy_next_q_values.argmax(dim=1, keepdim=True)
                    
                    
                    target_next_q_values = self.target_network(next_states)
                    next_q_values = target_next_q_values.gather(1, best_actions).squeeze(1)
                else:
                    
                    target_next_q_values = self.target_network(next_states)
                    next_q_values = target_next_q_values.max(1)[0]
                
                
                expected_q_values = rewards + \
                                   (1 - dones) * \
                                   self.config.discount_factor * \
                                   next_q_values
            
            
            loss = F.smooth_l1_loss(selected_q_values, expected_q_values)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            
            
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
            
           
            self.optimizer.step()
        
      
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        self.recent_losses.append(loss_value)
        
        
        self.total_steps += 1
        
       
        if self.total_steps % self.config.target_sync_frequency == 0:
            self.sync_target_network()
        
        return loss_value
    
    def sync_target_network(self):
       
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def save_checkpoint(self, filepath, episode=None):
       
        checkpoint = {
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode': episode,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
       
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return None
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        
       
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        
        self.total_steps = checkpoint['total_steps']
        
        return checkpoint.get('episode', None)
    
    def get_average_loss(self, window=100):
        
        if not self.recent_losses:
            return 0.0
        
        
        self.recent_losses = self.recent_losses[-window:]
        
        return np.mean(self.recent_losses)