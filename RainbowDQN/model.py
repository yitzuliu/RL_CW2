import math
import torch
from torch import nn
from torch.nn import functional as F

def scale_noise(size, device):
    noise_tensor = torch.randn(size, device=device)
    return noise_tensor.sign().mul_(noise_tensor.abs().sqrt_())

class NoisyLinear(nn.Module):
    def __init__(self, input_features, output_features, noise_std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.noise_std_init = noise_std_init

        self.mean_weight = nn.Parameter(torch.empty(output_features, input_features))
        self.std_weight = nn.Parameter(torch.empty(output_features, input_features))
        self.register_buffer('epsilon_weight', torch.empty(output_features, input_features))

        self.mean_bias = nn.Parameter(torch.empty(output_features))
        self.std_bias = nn.Parameter(torch.empty(output_features))
        self.register_buffer('epsilon_bias', torch.empty(output_features))

        self.initialize_parameters()
        self.sample_noise()

    def initialize_parameters(self):
        mu_range = 1 / math.sqrt(self.input_features)
        self.mean_weight.data.uniform_(-mu_range, mu_range)
        self.std_weight.data.fill_(self.noise_std_init / math.sqrt(self.input_features))
        self.mean_bias.data.uniform_(-mu_range, mu_range)
        self.std_bias.data.fill_(self.noise_std_init / math.sqrt(self.output_features))

    def sample_noise(self):
        input_noise = scale_noise(self.input_features, self.mean_weight.device)
        output_noise = scale_noise(self.output_features, self.mean_weight.device)
        self.epsilon_weight.copy_(output_noise.ger(input_noise))
        self.epsilon_bias.copy_(output_noise)

    def forward(self, input_tensor):
        if self.training:
            noisy_weight = self.mean_weight + self.std_weight * self.epsilon_weight
            noisy_bias = self.mean_bias + self.std_bias * self.epsilon_bias
            return F.linear(input_tensor, noisy_weight, noisy_bias)
        else:
            return F.linear(input_tensor, self.mean_weight, self.mean_bias)


class DQN(nn.Module):
    def __init__(self, args, num_actions):
        super(DQN, self).__init__()
        self.num_atoms = args.atoms
        self.num_actions = num_actions

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU()
        )

        self.convolutional_output_size = 3136

        self.value_hidden_layer = NoisyLinear(
            self.convolutional_output_size, args.hidden_size, noise_std_init=args.noisy_std
        )
        self.advantage_hidden_layer = NoisyLinear(
            self.convolutional_output_size, args.hidden_size, noise_std_init=args.noisy_std
        )

        self.value_output_layer = NoisyLinear(
            args.hidden_size, self.num_atoms, noise_std_init=args.noisy_std
        )
        self.advantage_output_layer = NoisyLinear(
            args.hidden_size, num_actions * self.num_atoms, noise_std_init=args.noisy_std
        )

    def forward(self, state_input, calculate_log=False):
        convolutional_features = self.convolutional_layers(state_input)
        flattened_features = convolutional_features.view(-1, self.convolutional_output_size)

        value_hidden = F.relu(self.value_hidden_layer(flattened_features))
        value_distribution_logits = self.value_output_layer(value_hidden)

        advantage_hidden = F.relu(self.advantage_hidden_layer(flattened_features))
        advantage_distribution_logits = self.advantage_output_layer(advantage_hidden)

        value_distribution_logits = value_distribution_logits.view(-1, 1, self.num_atoms)
        advantage_distribution_logits = advantage_distribution_logits.view(-1, self.num_actions, self.num_atoms)

        q_distribution_logits = value_distribution_logits + advantage_distribution_logits - advantage_distribution_logits.mean(1, keepdim=True)

        if calculate_log:
            output_log_probs = F.log_softmax(q_distribution_logits, dim=2)
            return output_log_probs
        else:
            output_probs = F.softmax(q_distribution_logits, dim=2)
            return output_probs

    def sample_new_noise_for_noisy_layers(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()

    def reset_noise(self):
        self.sample_new_noise_for_noisy_layers()
