import numpy as np
import torch
from utils import calculate_tree_start, propagate_index_segment_tree, propagate_segment_tree, retrieve_indices_from_tree
from utils import preprocess_state_for_storage, get_masked_transitions, calculate_batch_tensors, calculate_is_weights, get_validation_state

Transition_dtype = np.dtype([
    ('timestep', np.int32),
    ('state', np.uint8, (84, 84)),
    ('action', np.int32),
    ('reward', np.float32),
    ('nonterminal', np.bool_)
])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)



class SumTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.tree_start = calculate_tree_start(size)
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)
        self.max = 1.0 

    def update_tree_nodes(self, indices):
        propagate_segment_tree(self.sum_tree, indices)

    def propagate_tree_index(self, index):
        propagate_index_segment_tree(self.sum_tree, index)

    def update(self, indices, values):
        self.sum_tree[indices] = values
        self.update_tree_nodes(indices)
        current_max_value = np.max(values) if values.size > 0 else 1.0
        self.max = max(current_max_value, self.max)

    def update_single_index(self, index, value):
        self.sum_tree[index] = value
        self.propagate_tree_index(index)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data
        self.update_single_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def find(self, values):
        indices = retrieve_indices_from_tree(
            self.sum_tree, self.tree_start,
            np.zeros(values.shape, dtype=np.int32), values
        )
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight 
        self.priority_exponent = args.priority_exponent 
        self.t = 0 
        self.n_step_scaling = torch.tensor(
            [self.discount ** i for i in range(self.n)],
            dtype=torch.float32, device=self.device
        )
        self.transitions = SumTree(capacity)

    def append(self, state, action, reward, terminal):
        processed_state = preprocess_state_for_storage(state)
        self.transitions.append(
            (self.t, processed_state, action, reward, not terminal),
            self.transitions.max
        )
        self.t = 0 if terminal else self.t + 1

    def get_transitions_data(self, idxs):
        transition_idxs = np.arange(
            -self.history + 1, self.n + 1
        ) + np.expand_dims(idxs, axis=1)
        return self.transitions.get(transition_idxs)

    def sample_indices(self, batch_size):
        p_total = self.transitions.total()
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        while True:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            probs, data_idxs, tree_idxs = self.transitions.find(samples)

            history_condition = (data_idxs - self.transitions.index) % self.capacity >= self.history
            n_step_condition = (self.transitions.index - data_idxs) % self.capacity > self.n
            non_zero_prob = probs != 0

            if np.all(history_condition) and np.all(n_step_condition) and np.all(non_zero_prob):
                return probs, data_idxs, tree_idxs

    def sample(self, batch_size):
        probs, data_idxs, tree_idxs = self.sample_indices(batch_size)

        transitions_data = self.get_transitions_data(data_idxs)

        masked_transitions = get_masked_transitions(
            transitions_data, self.history, self.n, blank_trans
        )

        states, actions, returns, next_states, nonterminals = calculate_batch_tensors(
            masked_transitions, self.history, self.n, self.n_step_scaling, self.device
        )

        normalized_probs = probs / self.transitions.total()
        weights = calculate_is_weights(
            normalized_probs, self.capacity, self.transitions.index,
            self.transitions.full, self.priority_weight, self.device
        )

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities_alpha = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities_alpha)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.capacity:
             raise StopIteration
        if not self.transitions.full and self.current_idx >= self.transitions.index:
             raise StopIteration 
        indices = np.arange(self.current_idx - self.history + 1, self.current_idx + 1)
        transitions_data = self.transitions.get(indices)

        state = get_validation_state(
            transitions_data, self.history, blank_trans, self.device
        )

        self.current_idx += 1
        return state