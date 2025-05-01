import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
import ale_py
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def make_atari_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE, training=config.TRAINING_MODE, difficulty=config.DIFFICULTY):
    if training:
        render_mode = None
    env = gym.make(env_name, render_mode=render_mode, difficulty=difficulty, obs_type="rgb", frameskip=1, full_action_space=False)
    env = AtariPreprocessing(env, noop_max=config.NOOP_MAX, frame_skip=config.FRAME_SKIP, screen_size=config.FRAME_WIDTH, terminal_on_life_loss=True, grayscale_obs=True, grayscale_newaxis=True, scale_obs=False)
    env = FrameStackObservation(env, stack_size=config.FRAME_STACK)
    print(f"Environment: {env_name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    return env

if __name__ == "__main__":
    env = make_atari_env(render_mode="human", training=False, difficulty=0)
    observation, info = env.reset()
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space)
    print("Action meanings:", env.unwrapped.get_action_meanings())
    total_reward = 0
    step = 0
    while True:
        try:
            action = np.random.randint(1, env.action_space.n) if np.random.random() < 0.9 else 0
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if step % 20 == 0:
                print(f"Step {step}, Action: {action}, Reward: {reward}, Total: {total_reward}")
            if terminated or truncated:
                print(f"Episode finished after {step} steps with total reward: {total_reward}")
                observation, info = env.reset()
                total_reward = 0
                step = 0
        except KeyboardInterrupt:
            print("Test interrupted by user")
            break
    env.close()
