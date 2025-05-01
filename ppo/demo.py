import gymnasium as gym
import ale_py
import time
from ppo_cnn import PPO

dir_path = "ppo_MsPacman/hex_result"  # folder path

gym.register_envs(ale_py)

# build env
env = gym.make('ALE/MsPacman-v5',
               render_mode='human',
               difficulty=0,
               repeat_action_probability=0.0,
               full_action_space=False
               )

# init PPO agent
ppo = PPO(env)

# loading trained ppo weights
path = f"{dir_path}/weights"  # weights folder path
ppo.load(path)  
ppo.eval()

obs, _ = env.reset()
for _ in range(1000):
    action = ppo.choose_action(obs, epsilon=0)  # while testing, epsilon=0
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(0.02)  # lower speed for observation
    if terminated or truncated:
        break

env.close()
