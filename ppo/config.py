ep_max = 2000  # max training episodes
ep_len = 1000 # max steps in one episode

epsilon = 0.8
epsilon_decay = 0.995
epsilon_min = 0.05

actor_learning_rate = 1e-5
critic_learning_rate = 1e-4

gamma = 0.99
batch = 128
actor_uptaed_steps = 5
critic_uptaed_steps = 10

save_interval: int = 200  # interval for saving weights and logs