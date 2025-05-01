import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random
import time
import datetime
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.q_network import QNetwork
from src.per_memory import PERMemory
from src.device_utils import get_device


class DQNAgent:
    def __init__(self, state_shape, action_space_size,
                 learning_rate=config.LEARNING_RATE,
                 gamma=config.GAMMA,
                 epsilon_start=config.EPSILON_START,
                 epsilon_end=config.EPSILON_END,
                 epsilon_decay=config.EPSILON_DECAY,
                 memory_capacity=config.MEMORY_CAPACITY,
                 batch_size=config.BATCH_SIZE,
                 target_update_frequency=config.TARGET_UPDATE_FREQUENCY,
                 use_per=config.USE_PER,
                 per_log_frequency=config.PER_LOG_FREQUENCY,
                 evaluate_mode=config.DEFAULT_EVALUATE_MODE,
                 learning_starts=config.LEARNING_STARTS):

        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.use_per = use_per
        self.evaluate_mode = evaluate_mode
        self._epsilon = epsilon_start 
        self.per_log_frequency = per_log_frequency
        self.learning_starts = learning_starts

        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.policy_network = QNetwork(state_shape, action_space_size)
        self.target_network = QNetwork(state_shape, action_space_size)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        if use_per:
            self.memory = PERMemory(memory_capacity)
        else:
            self.memory = deque(maxlen=memory_capacity)
        
        self.steps_done = 0
        self.episode_rewards = []
        self.training_steps = 0
        
        self.loss_history = []
        self.epsilon_history = []
        self.reward_history = []
        self.priority_history = []
        self.learning_start_time = None
    
    @property
    def epsilon(self):
        return self._epsilon
    
    def select_action(self, state, evaluate=False):
        if isinstance(state, np.ndarray):
            if state.ndim == 3 and state.shape[-1] in [1, 3, 4]:
                state = np.transpose(state, (2, 0, 1))
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 3 else state.to(self.device)

        if evaluate:
            epsilon = 0
        else:
            if self.steps_done < self.learning_starts:
                epsilon = self.epsilon_start 
            else:
                progress = min(1.0, (self.steps_done - self.learning_starts) / self.epsilon_decay)
                if progress < 0.35:
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress) ** 2
                elif progress < 0.65:
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress)
                else:
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress) ** 0.5
            self.steps_done += 1
            if self.steps_done % 1000 == 0:
                self.epsilon_history.append((self.steps_done, epsilon))
        
        self._epsilon = epsilon
        
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_network(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_space_size)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        if self.use_per:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def _calculate_td_error(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        current_q = self.policy_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_state).max(1)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
        td_error = target_q - current_q
        return td_error.item()
    
    def optimize_model(self, logger=None):
        if self.use_per and len(self.memory) < self.batch_size:
            return 0.0
        elif not self.use_per and len(self.memory) < self.batch_size:
            return 0.0
        
        if self.learning_start_time is None:
            self.learning_start_time = time.time()
        
        if self.use_per:
            indices, weights, transitions = self.memory.sample(self.batch_size)
            batch_states = np.array([t.state for t in transitions])
            batch_actions = np.array([t.action for t in transitions])
            batch_rewards = np.array([t.reward for t in transitions])
            batch_next_states = np.array([t.next_state for t in transitions])
            batch_dones = np.array([t.done for t in transitions], dtype=np.float32)
            batch_weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = random.sample(self.memory, self.batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*transitions)
            batch_weights = torch.ones(self.batch_size).to(self.device)
        
        batch_states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        batch_actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
        batch_rewards = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        batch_dones = torch.FloatTensor(np.array(batch_dones)).to(self.device)
        
        current_q_values = self.policy_network(batch_states).gather(1, batch_actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_network(batch_next_states).max(1)[0]
        expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * next_q_values
        expected_q_values = expected_q_values.unsqueeze(1)
        td_errors = (expected_q_values - current_q_values).detach()
        
        if self.use_per:
            priorities = td_errors.abs().cpu().numpy().flatten()
            self.memory.update_priorities(indices, priorities)
            if self.training_steps % 100 == 0:
                self.priority_history.append((self.training_steps, priorities.mean()))
        
        element_wise_loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        batch_weights = batch_weights.view(-1, 1)
        weighted_loss = (batch_weights * element_wise_loss)
        loss = weighted_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), config.GRAD_CLIP_NORM)
        self.optimizer.step()
        
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            print(f"Target network updated at step {self.training_steps}")
        if self.training_steps % 100 == 0:
            self.loss_history.append((self.training_steps, loss.item()))
        if self.use_per:
            self.memory.update_beta(self.steps_done)
            if logger is not None and self.training_steps % self.per_log_frequency == 0:
                current_beta = self.memory.beta
                logger.log_per_update(
                    self.steps_done,
                    current_beta,
                    priorities,
                    td_errors.cpu().numpy().flatten(),
                    batch_weights.cpu().numpy().flatten()
                )
        return loss.item()
    
    def set_evaluation_mode(self, evaluate=True):
        self.evaluate_mode = evaluate
        if evaluate:
            self.policy_network.eval()
        else:
            self.policy_network.train()
    
    def save_model(self, path, save_optimizer=True, include_memory=False, metadata=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = {
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'steps_done': self.steps_done,
            'training_steps': self.training_steps,
            'epsilon': self._epsilon,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        if save_optimizer:
            model_state['optimizer'] = self.optimizer.state_dict()
        if include_memory and self.use_per:
            memory_state = self.memory.get_state_dict()
            model_state['memory'] = memory_state
        if metadata:
            model_state['metadata'] = metadata
        if hasattr(self, 'loss_history') and self.loss_history:
            model_state['metrics'] = {
                'loss_history': self.loss_history,
                'reward_history': self.reward_history if hasattr(self, 'reward_history') else [],
                'epsilon_history': self.epsilon_history if hasattr(self, 'epsilon_history') else [],
            }
        torch.save(model_state, path)
        print(f"Model saved to {path}")
        return True
    
    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file {path} not found.")
            return False
        try:
            model_state = torch.load(path, map_location=self.device)
            self.policy_network.load_state_dict(model_state['policy_network'])
            self.target_network.load_state_dict(model_state['target_network'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            self.steps_done = model_state['steps_done']
            self.training_steps = model_state['training_steps']
            self._epsilon = model_state['epsilon']
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
