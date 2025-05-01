import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir_path = "ppo_MsPacman/hex_result" # folder path
rewards = np.load(f"{dir_path}/raw_rewards.npy")
moving_avg_100 = np.load(f"{dir_path}/moving_avg_ep_reward_100.npy")
epsilons = np.load(f"{dir_path}/epsilon_record.npy")


# raw rewards chart
plt.plot(rewards, label="rewards")
plt.plot(pd.Series(rewards).rolling(100).mean(), label="moving avg (100)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Training Reward Curve")
plt.legend()
plt.show()

# moving avg(100) chart
plt.plot(moving_avg_100, label="moving avg(100)")
plt.xlabel("Episode")
plt.ylabel("Moving Avg Reward")
plt.title("PPO Training Moving Avg(100)")
plt.legend()
plt.show()

# epsilon chart
plt.plot(epsilons, label="epsilons")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("PPO Training Epsilon Curve")
plt.legend()
plt.show()