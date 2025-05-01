import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class SumTree:
    def __init__(self, memory_capacity=config.TREE_CAPACITY):
        self.memory_capacity = memory_capacity
        self.tree_size = 2 * memory_capacity - 1
        self.priority_tree = np.zeros(self.tree_size)
        self.experience_data = np.zeros(memory_capacity, dtype=object)
        self.next_write_index = 0
        self.experience_count = 0
    
    def _propagate_priority_change(self, tree_index, priority_change):
        parent_index = (tree_index - 1) // 2
        self.priority_tree[parent_index] += priority_change
        if parent_index != 0:
            self._propagate_priority_change(parent_index, priority_change)
    
    def _find_priority_leaf_index(self, tree_index, cumulative_value):
        left_child_index = 2 * tree_index + 1
        right_child_index = left_child_index + 1
        if left_child_index >= self.tree_size:
            return tree_index
        if cumulative_value <= self.priority_tree[left_child_index]:
            return self._find_priority_leaf_index(left_child_index, cumulative_value)
        else:
            return self._find_priority_leaf_index(right_child_index, cumulative_value - self.priority_tree[left_child_index])
    
    def total_priority(self):
        return self.priority_tree[0]
    
    def add(self, priority, experience_data):
        leaf_index = self.next_write_index + self.memory_capacity - 1
        self.experience_data[self.next_write_index] = experience_data
        self.update_priority(leaf_index, priority)
        self.next_write_index = (self.next_write_index + 1) % self.memory_capacity
        if self.experience_count < self.memory_capacity:
            self.experience_count += 1
    
    def update_priority(self, tree_index, new_priority):
        priority_change = new_priority - self.priority_tree[tree_index]
        self.priority_tree[tree_index] = new_priority
        self._propagate_priority_change(tree_index, priority_change)
    
    def get_experience_by_priority(self, priority_value):
        leaf_index = self._find_priority_leaf_index(0, priority_value)
        data_index = leaf_index - self.memory_capacity + 1
        return (leaf_index, self.priority_tree[leaf_index], self.experience_data[data_index])
    
    def get_all_priorities(self):
        return self.priority_tree[self.memory_capacity - 1:self.memory_capacity - 1 + self.experience_count]

