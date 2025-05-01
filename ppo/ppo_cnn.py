import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import os
from config import actor_learning_rate, critic_learning_rate, actor_uptaed_steps, critic_uptaed_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()

        action_dim = env.action_space.n

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # (3, 210, 160) -> (32, 51, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32, 51, 39) -> (64, 24, 18)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 24, 18) -> (64, 22, 16)
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 210, 160)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Actor Network
        self.actor_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

        # Critic Network
        self.critic_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        # Old Policy Network
        self.old_actor_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(list(self.cnn.parameters()) + list(self.actor_fc.parameters()), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(list(self.cnn.parameters()) + list(self.critic_fc.parameters()), lr=critic_learning_rate)

        self.to(device)

    def extract_features(self, state):
        if isinstance(state, np.ndarray):
            if state.ndim == 3:
                state = np.transpose(state, (2, 0, 1))
                state = state[np.newaxis, ...]
            elif state.ndim == 4:
                state = np.transpose(state, (0, 3, 1, 2))
            state = torch.tensor(state, dtype=torch.float32, device=device) / 255.0
        elif torch.is_tensor(state):
            if state.ndim == 3:
                state = state.permute(2, 0, 1).unsqueeze(0)
            elif state.ndim == 4:
                state = state.permute(0, 3, 1, 2)
            state = state.float().to(device) / 255.0
        else:
            raise ValueError("state must be np.ndarray or torch.Tensor")
        return self.cnn(state)

    def get_action_dist(self, state):  
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)
        
        x = self.extract_features(state)
        logits = self.actor_fc(x)
        return dist.Categorical(logits=logits)

    def get_old_action_dist(self, state):
        x = self.extract_features(state)
        logits = self.old_actor_fc(x)
        return dist.Categorical(logits=logits)

    def get_value(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        x = self.extract_features(state)
        return self.critic_fc(x)

    def update_old_policy(self):
        self.old_actor_fc.load_state_dict(self.actor_fc.state_dict())

    def train_critic(self, states, discounted_rewards):
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device)
        else:
            states = states.to(device)
        if isinstance(discounted_rewards, np.ndarray):
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        else:
            discounted_rewards = discounted_rewards.to(device)

        self.critic_optimizer.zero_grad()
        values = self.get_value(states)
        loss = nn.MSELoss()(values.squeeze(), discounted_rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        return {
            'critic_loss': loss.item(),
            'value_mean': values.mean().item(),
            'value_std': values.std().item()
        }

    def train_actor(self, states, actions, advantages, epsilon, entropy_coef=0.5):
        # ensure trainging on GPU
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=device)
        else:
            states = states.to(device)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.int64, device=device)
        else:
            actions = actions.to(device)
        if isinstance(advantages, np.ndarray):
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        else:
            advantages = advantages.to(device)
        
        self.actor_optimizer.zero_grad()
        pi = self.get_action_dist(states) # current policy
        oldpi = self.get_old_action_dist(states) # old policy
        log_prob = pi.log_prob(actions)
        old_log_prob = oldpi.log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob) # calculate the ratio of probabilities
        surr1 = ratio * advantages # unclipped surrogate loss
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages # clipped surrogate loss
        entropy = pi.entropy().mean()
        loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy # calculate the final loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': loss.item(),
            'entropy': entropy.item()
        }

    def choose_action(self, state, epsilon=0.2):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        if np.random.rand() < epsilon:
            return np.random.randint(self.actor_fc[-1].out_features)
        dist = self.get_action_dist(state)
        action = dist.sample().item()
        return action

    def get_v(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        value = self.get_value(state)
        return value.item()

    def update(self, states, actions, rewards, epsilon):
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        self.update_old_policy()
        values = self.get_value(states).squeeze()
        advantages = (rewards - values).detach()
        
        actor_metrics = []
        critic_metrics = []
        
        for _ in range(actor_uptaed_steps):
            metrics = self.train_actor(states, actions, advantages, epsilon)
            actor_metrics.append(metrics)
        
        for _ in range(critic_uptaed_steps):
            metrics = self.train_critic(states, rewards)
            critic_metrics.append(metrics)
            
        avg_metrics = {
            'actor_loss': np.mean([m['actor_loss'] for m in actor_metrics]),
            'critic_loss': np.mean([m['critic_loss'] for m in critic_metrics]),
            'entropy': np.mean([m['entropy'] for m in actor_metrics]),
        }
        
        return avg_metrics

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.cnn.state_dict(), f"{path}/cnn.pth")
        torch.save(self.actor_fc.state_dict(), f"{path}/actor_fc.pth")
        torch.save(self.critic_fc.state_dict(), f"{path}/critic_fc.pth")
        torch.save(self.old_actor_fc.state_dict(), f"{path}/old_actor_fc.pth")

    def load(self, path):
        self.cnn.load_state_dict(torch.load(f"{path}/cnn.pth", map_location='cpu'))
        self.actor_fc.load_state_dict(torch.load(f"{path}/actor_fc.pth", map_location='cpu'))
        self.critic_fc.load_state_dict(torch.load(f"{path}/critic_fc.pth", map_location='cpu'))
        self.old_actor_fc.load_state_dict(torch.load(f"{path}/old_actor_fc.pth", map_location='cpu'))