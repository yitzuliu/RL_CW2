import os
import numpy as np
import torch
from utils import create_dqn_network, load_and_adapt_state_dict, setup_target_network, setup_optimizer, calculate_expected_q, calculate_target_distribution
from utils import compute_loss, optimize_model


class Agent:
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(
            args.V_min, args.V_max, self.atoms, device=args.device
        )
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.device = args.device

        self.online_net = create_dqn_network(args, self.action_space, self.device)
        if args.trained_model:
            state_dict = load_and_adapt_state_dict(args.trained_model)
            self.online_net.load_state_dict(state_dict)
            print("Successfully loaded pretrained model.")
        self.online_net.train()

        self.target_net = setup_target_network(
            self.online_net, args, self.action_space, self.device
        )

        self.optimiser = setup_optimizer(
            self.online_net.parameters(), args.learning_rate, args.adam_eps
        )

    def reset_noise(self):
        self.online_net.reset_noise()

    def act(self, state):
        with torch.no_grad():
            distribution = self.online_net(state.unsqueeze(0))
            q_values = calculate_expected_q(distribution, self.support)
            return q_values.argmax(1).item()

    def act_e_greedy(self, state, epsilon=0.001):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)
        else:
            return self.act(state)

    def learn(self, mem):
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        target_distribution_m = calculate_target_distribution(
            self.online_net, self.target_net, next_states, returns, nonterminals,
            self.support, self.delta_z, self.discount, self.n,
            self.Vmin, self.Vmax, self.batch_size, self.atoms, self.device
        )

        log_ps = self.online_net(states, calculate_log=True)
        log_ps_a = log_ps[range(self.batch_size), actions]

        loss = compute_loss(log_ps_a, target_distribution_m)

        optimize_model(loss, weights, self.online_net, self.optimiser, self.norm_clip)

        mem.update_priorities(idxs, loss.detach().cpu().numpy())

        return loss.mean().item() 

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def evaluate_q(self, state):
        with torch.no_grad():
            distribution = self.online_net(state.unsqueeze(0))
            q_values = calculate_expected_q(distribution, self.support)
            return q_values.max(1)[0].item()

    def get_current_lr(self):
        return self.optimiser.param_groups[0]['lr']

    def reduce_learning_rate(self, factor=0.5):
        for param_group in self.optimiser.param_groups:
            param_group['lr'] *= factor
        return self.get_current_lr()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
