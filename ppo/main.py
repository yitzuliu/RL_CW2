import numpy as np
import gymnasium as gym
import ale_py
from config import ep_max, ep_len, epsilon, epsilon_decay, epsilon_min, gamma, batch, save_interval
from ppo_cnn import PPO
import os
import sys
import datetime
import glob
import shutil
from collections import deque

# only used when training in hex
def clean_old_logs(keep=5):
    logs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "log_*.txt")))
    if len(logs) > keep:
        # if there is no enough space, only keep the fisrt and last (keep-1)
        logs_to_keep = [logs[0]] + logs[-(keep-1):]
        logs_to_remove = [log for log in logs if log not in logs_to_keep]
        for log in logs_to_remove:
            os.remove(log)
            print(f"Removed old log: {log}", flush=True)

# only used when training in hex
def check_disk_space_and_clean(threshold_gb=1, keep_logs=5):
    total, used, free = shutil.disk_usage(os.path.dirname(__file__))
    if free // (1024**3) < threshold_gb:
        print("Warning: Disk space low! Auto-cleaning old logs...", flush=True)
        clean_old_logs(keep=keep_logs)


# setting environment
gym.register_envs(ale_py)
env = gym.make("ALE/MsPacman-v5",
               render_mode=None,
               difficulty=0,
               repeat_action_probability=0,
               full_action_space=False,
               frameskip=8)

ppo = PPO(env)

raw_ep_reward = [] # list for episode rewards
moving_avg_ep_reward = [] # list for moving average rewards（moving_avg_ep_reward[-1] * 0.9 + ep_reward * 0.1)
episode_rewards = deque(maxlen=100)  # list for last 100 episode rewards
moving_avg_ep_reward_100 = [] # list for moving average of last 100 episode rewards
epsilon_record = []
actor_loss_record = []
critic_loss_record = []


original_stdout = sys.stdout  # save original stdout
log_file = None # Initialize log file variable

# Run training loop
try:
    for ep in range(ep_max):
        
        # used for saving log file in hex
        if ep % save_interval == 0:
            check_disk_space_and_clean(threshold_gb=1, keep_logs=5)
            if log_file is not None:
                log_file.close()
            log_path = os.path.join(os.path.dirname(__file__), f"log_{ep:04d}_{ep+save_interval-1:04d}.txt")
            log_file = open(log_path, "w", buffering=1)
            sys.stdout = log_file
            sys.stderr = log_file

        result = env.reset()
        state, _ = result if isinstance(result, tuple) else (result, None)

        buffer_state, buffer_action, buffer_reward = [], [], []
        ep_reward = 0
        final_reward = 0
        final_terminated = False
        episode_actions = []  # record all actions in this episode
        steps = 0

        # giving fixed epsilon in the beginning to explore more and then decay
        if ep < 50:
            epsilon = 0.8
        else:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        for t in range(ep_len): # steps in one episode
            
            action = ppo.choose_action(state, epsilon=epsilon)  # choose action
            episode_actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append(reward)

            state = next_state
            ep_reward += reward
            steps += 1

            if done:
                final_reward = reward
                final_terminated = terminated
                break

            # update ppo
            if (t+1) % batch == 0 or t == ep_len-1:
                next_state_value = ppo.get_v(next_state)

                assert len(buffer_state) == len(buffer_action) == len(buffer_reward), "❌ Buffer length mismatch!"

                discounted_reward = []
                for reward in buffer_reward[::-1]:
                    next_state_value = reward + gamma * next_state_value
                    discounted_reward.append(next_state_value)
                discounted_reward.reverse()

                bs = np.array(buffer_state)
                ba = np.vstack(buffer_action)
                br = np.array(discounted_reward)[:, np.newaxis]
                
                # clear the buffer
                buffer_state, buffer_action, buffer_reward = [], [], []

                metrics = ppo.update(bs, ba, br, epsilon)
                actor_loss = metrics['actor_loss']
                critic_loss = metrics['critic_loss']
                entropy = metrics['entropy']
        
        # record the training results and params        
        raw_ep_reward.append(ep_reward)

        if ep == 0:
            moving_avg_ep_reward.append(ep_reward)
        else:
            moving_avg_ep_reward.append(moving_avg_ep_reward[-1] * 0.9 + ep_reward * 0.1)

        episode_rewards.append(ep_reward)
        avg_r = sum(episode_rewards) / len(episode_rewards)
        moving_avg_ep_reward_100.append(avg_r) # average max 100
        epsilon_record.append(epsilon)
        actor_loss_record.append(actor_loss)
        critic_loss_record.append(critic_loss)

        # print episode info
        print(f"\nEpisode {ep}, Reward: {ep_reward}, MovingAvg({len(episode_rewards)}): {avg_r:6.2f}, avgMovingReward: {moving_avg_ep_reward[-1]:.1f}, Epsilon: {epsilon:.4f}, Actor_Loss: {actor_loss:.4f}, Critic_Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}", flush=True)
        

        # save model and records every save_interval episodes
        if (ep + 1) % save_interval == 0:
            save_dir = os.path.join(os.path.dirname(__file__), f"weights")
            ppo.save(save_dir)

            dir_path = os.path.dirname(__file__)          
            np.save(f"{dir_path}/raw_rewards.npy", np.array(raw_ep_reward))
            np.save(f"{dir_path}/moving_avg_ep_reward.npy", np.array(moving_avg_ep_reward))
            np.save(f"{dir_path}/moving_avg_ep_reward_100.npy", np.array(moving_avg_ep_reward_100))
            np.save(f"{dir_path}/epsilon_record.npy", np.array(epsilon_record))
            np.save(f"{dir_path}/actor_loss_record.npy", np.array(actor_loss_record))
            np.save(f"{dir_path}/critic_loss_record.npy", np.array(critic_loss_record))
            
            print(f"Checkpoint saved at episode {ep+1} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", flush=True)

    print("train finished!", flush=True)

finally:
    # used for training in hex
    if log_file is not None:
        log_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stdout