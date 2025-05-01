import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir_path = "ppo_MsPacman/hex_result" # folder path
rewards = np.load(f"{dir_path}/raw_rewards.npy")
moving_avg = np.load(f"{dir_path}/moving_avg_ep_reward.npy")
moving_avg_100 = np.load(f"{dir_path}/moving_avg_ep_reward_100.npy")
epsilons = np.load(f"{dir_path}/epsilon_record.npy")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# raw rewards chart
axes[0, 0].plot(rewards, label="raw")
axes[0, 0].plot(pd.Series(rewards).rolling(100).mean(), label="moving avg (100)")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].set_title("Raw & Rolling Averages")
axes[0, 0].legend()

# moving avg(100) chart
axes[0, 1].plot(moving_avg_100, label="moving_avg_100")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Moving Avg Reward")
axes[0, 1].set_title("Precomputed Moving Avg (100)")
axes[0, 1].legend()

# epsilon chart
axes[1, 0].plot(epsilons, label="episode")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Epsilon")
axes[1, 0].set_title("PPO Training Epsilon Curve")
axes[1, 0].legend()

# moving avg chart
axes[1, 1].plot(moving_avg, label="moving avg")
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("moving_avg")
axes[1, 1].set_title("Moving Avg")
axes[1, 1].legend()

plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()