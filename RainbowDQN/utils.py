from datetime import datetime
from model import DQN
import os
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import cv2


def log_on_terminal(str, importance=0):
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        if importance > 0:
            if importance == 1:
                print(f'[{timestamp}] \033[93m{str}\033[0m')
            else:
                print(f'[{timestamp}] \033[91m{str}\033[0m')
        else:
            print(f'[{timestamp}] {str}')





def create_dqn_network(args, action_space, device):
    return DQN(args, action_space).to(device=device)

def load_and_adapt_state_dict(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Pretrained model file not found: {model_path}")

    print(f"Loading pretrained model: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')

    if 'conv1.weight' in state_dict:
        print("Adapting state dictionary from older model format...")
        key_map = {
            'conv1.weight': 'convs.0.weight', 'conv1.bias': 'convs.0.bias',
            'conv2.weight': 'convs.2.weight', 'conv2.bias': 'convs.2.bias',
            'conv3.weight': 'convs.4.weight', 'conv3.bias': 'convs.4.bias'
        }
        for old_key, new_key in key_map.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
    return state_dict

def setup_target_network(online_net, args, action_space, device):
    target_net = create_dqn_network(args, action_space, device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.train()
    for param in target_net.parameters():
        param.requires_grad = False
    return target_net

def setup_optimizer(online_net_parameters, lr, adam_eps):
    return optim.Adam(online_net_parameters, lr=lr, eps=adam_eps)

def calculate_target_distribution(
    online_net, target_net, next_states, returns, nonterminals,
    support, atom_spacing, discount, n_steps, value_min, value_max, batch_size, num_atoms, device
):
    with torch.no_grad():
        next_state_probs_online = online_net(next_states)
        next_state_values_components = support.expand_as(next_state_probs_online) * next_state_probs_online
        next_state_q_values = next_state_values_components.sum(2) 
        best_action_indices = next_state_q_values.argmax(1)

        target_net.reset_noise()
        next_state_probs_target = target_net(next_states)
        target_probs_for_best_actions = next_state_probs_target[range(batch_size), best_action_indices]

        projected_values = returns.unsqueeze(1) + nonterminals * (discount**n_steps) * support.unsqueeze(0)
        projected_values = projected_values.clamp(min=value_min, max=value_max)
        
        bin_positions = (projected_values - value_min) / atom_spacing
        lower_bin_indices = bin_positions.floor().to(torch.int64)
        upper_bin_indices = bin_positions.ceil().to(torch.int64)

        lower_bin_indices[(upper_bin_indices > 0) * (lower_bin_indices == upper_bin_indices)] -= 1
        upper_bin_indices[(lower_bin_indices < (num_atoms - 1)) * (lower_bin_indices == upper_bin_indices)] += 1

        target_distribution = torch.zeros(batch_size, num_atoms, device=device)
        batch_atom_offsets = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, num_atoms).to(lower_bin_indices)
        
        target_distribution.view(-1).index_add_(
            0, 
            (lower_bin_indices + batch_atom_offsets).view(-1), 
            (target_probs_for_best_actions * (upper_bin_indices.float() - bin_positions)).view(-1)
        )
        target_distribution.view(-1).index_add_(
            0, 
            (upper_bin_indices + batch_atom_offsets).view(-1), 
            (target_probs_for_best_actions * (bin_positions - lower_bin_indices.float())).view(-1)
        )
    return target_distribution


def compute_loss(log_ps_a, target_distribution_m):
    return -torch.sum(target_distribution_m * log_ps_a, 1)

def optimize_model(loss, weights, online_net, optimizer, norm_clip):
    optimizer.zero_grad()
    weighted_loss = (weights * loss).mean() 
    weighted_loss.backward()
    clip_grad_norm_(online_net.parameters(), norm_clip)
    optimizer.step()

def calculate_expected_q(distribution, support):
    return (distribution * support).sum(2)



def process_frame(observation, device):
    state = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
    return torch.tensor(state, dtype=torch.float32, device=device).div_(255)

def reshape_reward_pacman(reward_sum, died, level_finished):
    ghost_rewards = {200, 400, 800, 1600}
    
    reward = 0.0
    
    reward += 0.01
    
    if reward_sum == 10: 
        reward += 0.5
    elif reward_sum == 50: 
        reward += 1.0  
    elif reward_sum in ghost_rewards:
        reward += 1.5
    elif reward_sum >= 100 and reward_sum not in ghost_rewards: 
        reward += 0.5
    
    if level_finished:
        reward += 4.0
    if died:
        reward -= 1.0  
        
    return reward


def perform_frame_skip(env, action_map, action, frame_processor, device):
    frame_buffer = torch.zeros(2, 84, 84, device=device)
    raw_reward_sum = 0
    done = False
    truncated = False
    info = {}

    for t in range(4):
        observation, reward, terminated, truncated, info = env.step(
            action_map[action]
        )
        raw_reward_sum += reward

        if t == 2:
            frame_buffer[0] = frame_processor(observation, device)
        elif t == 3:
            frame_buffer[1] = frame_processor(observation, device)

        done = terminated or truncated
        if done:
            break

    max_pooled_observation = frame_buffer.max(0)[0]
    return max_pooled_observation, raw_reward_sum, done, info





def calculate_tree_start(size):
    return 2**(size - 1).bit_length() - 1

def propagate_segment_tree(sum_tree, indices):
    valid_indices = indices[ (indices >= 0) & (indices < sum_tree.shape[0]) ]
    if valid_indices.size == 0:
        return 

    parents = (valid_indices - 1) // 2
    unique_parents = np.unique(parents[parents >= 0])

    if unique_parents.size == 0:
        return 

    parents_updated = False
    for parent_idx in unique_parents:
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = 2 * parent_idx + 2

        left_sum = sum_tree[left_child_idx] if left_child_idx < sum_tree.shape[0] else 0
        right_sum = sum_tree[right_child_idx] if right_child_idx < sum_tree.shape[0] else 0

        new_sum = left_sum + right_sum

        if sum_tree[parent_idx] != new_sum:
            sum_tree[parent_idx] = new_sum
            parents_updated = True

    if parents_updated:
        propagate_segment_tree(sum_tree, unique_parents)


def propagate_index_segment_tree(sum_tree, index):
    parent = (index - 1) // 2
    if parent < 0: 
        return
    left, right = 2 * parent + 1, 2 * parent + 2
    left_val = sum_tree[left] if left < sum_tree.shape[0] else 0
    right_val = sum_tree[right] if right < sum_tree.shape[0] else 0
    sum_tree[parent] = left_val + right_val
    if parent != 0:
        propagate_index_segment_tree(sum_tree, parent)

def retrieve_indices_from_tree(sum_tree, tree_start, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))

    if children_indices[0, 0] >= sum_tree.shape[0]:
        return indices

    if children_indices[0, 0] >= tree_start:
         children_indices = np.minimum(children_indices, sum_tree.shape[0] - 1)

    left_children_values = sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)
    successor_indices = children_indices[successor_choices, np.arange(indices.size)]
    successor_values = values - successor_choices * left_children_values
    return retrieve_indices_from_tree(sum_tree, tree_start, successor_indices, successor_values)


def preprocess_state_for_storage(state):
    return state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))

def get_masked_transitions(transitions_data, history_len, n_steps, blank_trans):
    transitions = np.copy(transitions_data)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)

    for t in range(history_len - 2, -1, -1):
        blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])

    for t in range(history_len, history_len + n_steps):
        blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])

    transitions[blank_mask] = blank_trans
    return transitions

def calculate_batch_tensors(masked_transitions, history_len, n_steps, n_step_scaling, device):
    all_states = masked_transitions['state']
    states = torch.tensor(all_states[:, :history_len].copy(), device=device, dtype=torch.float32).div_(255)
    next_states = torch.tensor(all_states[:, n_steps : n_steps + history_len].copy(), device=device, dtype=torch.float32).div_(255)

    actions = torch.tensor(masked_transitions['action'][:, history_len - 1].copy(), dtype=torch.int64, device=device)

    reward_start_idx = history_len - 1
    reward_end_idx = history_len + n_steps - 1
    rewards = torch.tensor(masked_transitions['reward'][:, reward_start_idx : reward_end_idx].copy(), dtype=torch.float32, device=device)
    if rewards.shape[1] == n_step_scaling.shape[0]:
         R = torch.matmul(rewards, n_step_scaling)
    elif rewards.shape[1] == 0 and n_step_scaling.shape[0] == 0: 
         R = torch.zeros(rewards.shape[0], device=device, dtype=torch.float32)
    else:
        print(f"Warning: Reward shape mismatch during n-step calculation. Rewards shape: {rewards.shape}, Scaling shape: {n_step_scaling.shape}")
        R = torch.zeros(rewards.shape[0], device=device, dtype=torch.float32)

    nonterminal_idx = history_len + n_steps - 1
    nonterminals = torch.tensor(np.expand_dims(masked_transitions['nonterminal'][:, nonterminal_idx].copy(), axis=1), dtype=torch.float32, device=device)

    return states, actions, R, next_states, nonterminals

def calculate_is_weights(probs, capacity, current_index, is_full, priority_weight_beta, device):
    current_capacity = capacity if is_full else current_index
    weights = (current_capacity * probs) ** -priority_weight_beta
    normalized_weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=device)
    return normalized_weights

def get_validation_state(transitions_data, history_len, blank_trans, device):
    transitions = np.copy(transitions_data)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(history_len - 1)):
        blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])
    transitions[blank_mask] = blank_trans
    state = torch.tensor(transitions['state'], dtype=torch.float32, device=device).div_(255)
    return state


