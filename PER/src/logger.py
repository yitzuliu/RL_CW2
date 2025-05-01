import os
import json
import time
import datetime
import numpy as np
from collections import deque
import sys
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class Logger:
    def __init__(self, log_dir: str = config.LOG_DIR, 
                 data_dir: str = config.DATA_DIR,
                 experiment_name: str = None,
                 enable_file_logging: bool = config.ENABLE_FILE_LOGGING,
                 save_interval: int = config.LOGGER_SAVE_INTERVAL, 
                 memory_window: int = config.LOGGER_MEMORY_WINDOW,
                 batch_size: int = config.LOGGER_BATCH_SIZE,
                 per_log_frequency: int = config.PER_LOG_FREQUENCY):
        if experiment_name is None:
            self.experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        self.data_dir = os.path.join(data_dir, self.experiment_name)
        for directory in [self.log_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
        self.save_interval = save_interval
        self.memory_window = memory_window
        self.batch_size = batch_size
        self.per_log_frequency = per_log_frequency
        self.enable_file_logging = enable_file_logging
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilon_values: List[float] = []
        self.beta_values: List[Tuple[int, float]] = []
        self.priority_means: List[Tuple[int, float]] = []
        self.priority_maxes: List[Tuple[int, float]] = []
        self.td_error_means: List[Tuple[int, float]] = []
        self.is_weight_means: List[Tuple[int, float]] = []
        self.last_logged_beta = None
        self.last_logged_epsilon = None
        self.total_steps: int = 0
        self.current_episode: int = 0
        self.start_time = time.time()
        self.episode_start_time = time.time()
        self.best_eval_reward: float = float('-inf')
        self.reward_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=100)
        self.data_buffer = []
        self.buffer_count = 0
        self.per_data_buffer = []
        self.per_buffer_count = 0
        self.log_file_path = os.path.join(self.log_dir, "training_log.txt")
        self.episode_data_path = os.path.join(self.data_dir, "episode_data.jsonl")
        self.per_data_path = os.path.join(self.data_dir, "per_data.jsonl")
        if enable_file_logging:
            with open(self.log_file_path, "w") as f:
                f.write(f"Training Log - Experiment: {self.experiment_name}\n")
                f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        self.log_text(f"Logger initialized. Experiment: {self.experiment_name}")
        
    def log_episode_start(self, episode_num: int):
        self.current_episode = episode_num
        self.episode_start_time = time.time()
    
    def log_episode_end(self, episode_num: int, 
                        total_reward: float, 
                        steps: int, 
                        avg_loss: Optional[float] = None,
                        epsilon: Optional[float] = None,
                        beta: Optional[float] = None):
        duration = time.time() - self.episode_start_time
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.reward_window.append(total_reward)
        if avg_loss is not None:
            self.episode_losses.append(avg_loss)
            self.loss_window.append(avg_loss)
        self.total_steps += steps
        avg_reward_100 = np.mean(self.reward_window) if self.reward_window else 0.0
        avg_loss_100 = np.mean(self.loss_window) if self.loss_window and avg_loss is not None else 0.0
        episode_data = {
            "episode": episode_num,
            "reward": total_reward,
            "steps": steps,
            "duration": duration,
            "avg_reward_100": avg_reward_100,
            "timestamp": time.time()
        }
        if avg_loss is not None:
            episode_data["loss"] = avg_loss
            episode_data["avg_loss_100"] = avg_loss_100
        if epsilon is not None:
            episode_data["epsilon"] = epsilon
        if beta is not None:
            episode_data["beta"] = beta
        self.data_buffer.append(episode_data)
        self.buffer_count += 1
        loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
        epsilon_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
        beta_str = f"{beta:.4f}" if beta is not None else "N/A"
        log_line = f"episode {episode_num} | steps {steps} | reward {total_reward:.2f} | loss {loss_str}"
        if epsilon is not None:
            log_line += f" | Epsilon {epsilon_str}"
        if beta is not None:
            log_line += f" | Beta {beta_str}"
        log_line += f" | during {duration:.2f}s"
        print(log_line)
        if self.enable_file_logging:
            try:
                with open(self.log_file_path, "a") as f:
                    f.write(f"{log_line}\n")
            except OSError as e:
                print(f"WARNING: Failed to write to log file: {str(e)}. Continuing without logging to file.")
        if episode_num % config.LOGGER_DETAILED_INTERVAL == 0:
            self.log_text(self._format_progress_report(detailed=True))
        if self.buffer_count >= self.batch_size:
            self._batch_write()
        self.limit_memory_usage()
    
    def log_step(self, step_num: int, reward: float, loss: Optional[float] = None):
        pass
    
    def log_per_update(self, step_num: int, 
                      beta: float, 
                      priorities: np.ndarray, 
                      td_errors: np.ndarray,
                      is_weights: np.ndarray):
        significant_beta_change = (
            self.last_logged_beta is None or 
            abs(beta - self.last_logged_beta) > 0.01 or
            step_num % (self.per_log_frequency * 10) == 0
        )
        if significant_beta_change:
            self.beta_values.append((step_num, beta))
            self.last_logged_beta = beta
        mean_priority = float(np.mean(priorities))
        max_priority = float(np.max(priorities)) 
        mean_td_error = float(np.mean(np.abs(td_errors)))
        mean_is_weight = float(np.mean(is_weights))
        self.priority_means.append((step_num, mean_priority))
        self.priority_maxes.append((step_num, max_priority))
        self.td_error_means.append((step_num, mean_td_error))
        self.is_weight_means.append((step_num, mean_is_weight))
        per_data = {
            "step": step_num,
            "beta": beta,
            "mean_priority": mean_priority,
            "max_priority": max_priority,
            "mean_td_error": mean_td_error,
            "mean_is_weight": mean_is_weight,
            "timestamp": time.time()
        }
        self.per_data_buffer.append(per_data)
        self.per_buffer_count += 1
        if step_num % (self.per_log_frequency * 20) == 0:
            self.log_text(
                f"PER Update - Step: {step_num}, Beta: {beta:.4f}, "
                f"Mean Priority: {mean_priority:.6f}, Max Priority: {max_priority:.6f}"
            )
        if self.per_buffer_count >= config.PER_BATCH_SIZE or (self.per_buffer_count >= config.PER_BATCH_SIZE / 2 and step_num % (self.per_log_frequency * 5) == 0):
            self._batch_write_per()

    def log_epsilon(self, step_num: int, epsilon: float):
        significant_epsilon_change = (
            self.last_logged_epsilon is None or 
            abs(epsilon - self.last_logged_epsilon) > 0.01 or
            step_num % 5000 == 0
        )
        if significant_epsilon_change:
            self.epsilon_values.append(epsilon)
            self.last_logged_epsilon = epsilon
    
    def limit_memory_usage(self):
        if len(self.episode_rewards) > self.memory_window:
            trim_count = len(self.episode_rewards) - self.memory_window
            self.episode_rewards = self.episode_rewards[trim_count:]
            self.episode_lengths = self.episode_lengths[trim_count:]
            self.episode_losses = self.episode_losses[trim_count:]
            self.epsilon_values = self.epsilon_values[trim_count:]
        max_per_records = self.memory_window * 5
        for metric_list in [self.beta_values, self.priority_means, 
                           self.priority_maxes, self.td_error_means, self.is_weight_means]:
            if len(metric_list) > max_per_records:
                if len(metric_list) > max_per_records * 2:
                    first_n = max(10, int(max_per_records * 0.1))
                    last_n = max(30, int(max_per_records * 0.3))
                    middle_n = max_per_records - first_n - last_n
                    middle_start = first_n
                    middle_end = len(metric_list) - last_n
                    middle_indices = np.linspace(middle_start, middle_end-1, middle_n, dtype=int)
                    metric_list[:] = (
                        metric_list[:first_n] + 
                        [metric_list[i] for i in middle_indices] + 
                        metric_list[-last_n:]
                    )
                else:
                    metric_list[:] = metric_list[-max_per_records:]
    
    def get_training_data(self, metric_name=None, start=None, end=None):
        self._batch_write()
        self._batch_write_per()
        data = {
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
            "losses": self.episode_losses,
            "epsilon_values": self.epsilon_values,
            "beta_values": self.beta_values,
            "priority_means": self.priority_means,
            "priority_maxes": self.priority_maxes,
            "td_error_means": self.td_error_means,
            "is_weight_means": self.is_weight_means
        }
        if metric_name and metric_name in data:
            metric_data = data[metric_name]
            if start is not None or end is not None:
                start = start or 0
                end = end or len(metric_data)
                return metric_data[start:end]
            return metric_data
        if start is not None or end is not None:
            start = start or 0
            end = end or len(self.episode_rewards)
            sliced_data = data.copy()
            sliced_data["rewards"] = data["rewards"][start:end]
            sliced_data["lengths"] = data["lengths"][start:end]
            sliced_data["losses"] = data["losses"][start:end]
            return sliced_data
        return data
    
    def get_training_summary(self):
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        current_reward_avg = np.mean(self.reward_window) if self.reward_window else 0
        best_reward = max(self.episode_rewards) if self.episode_rewards else 0
        best_episode = self.episode_rewards.index(best_reward) + 1 if self.episode_rewards else 0
        summary = {
            "experiment_name": self.experiment_name,
            "episodes_completed": self.current_episode,
            "total_steps": self.total_steps,
            "duration": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
            "current_reward_avg": current_reward_avg,
            "best_reward": best_reward,
            "best_episode": best_episode,
            "best_eval_reward": self.best_eval_reward,
            "last_rewards": self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
        return summary
        
    def log_text(self, message: str):
        print(message)
        if self.enable_file_logging:
            try:
                with open(self.log_file_path, "a") as f:
                    f.write(message + "\n")
            except OSError as e:
                print(f"WARNING: Failed to write to log file: {str(e)}. Continuing without logging to file.")
    
    def update_best_eval_reward(self, new_best_reward):
        if new_best_reward > self.best_eval_reward:
            self.best_eval_reward = new_best_reward
            self.log_text(f"Updated best evaluation reward to: {new_best_reward:.2f}")
    
    def _batch_write(self):
        if not self.data_buffer:
            return
        try:
            with open(self.episode_data_path, 'a') as f:
                for record in self.data_buffer:
                    f.write(json.dumps(record) + '\n')
        except OSError as e:
            print(f"WARNING: Failed to write to episode data file: {str(e)}. Continuing without logging to file.")
        self.data_buffer = []
        self.buffer_count = 0
    
    def _batch_write_per(self):
        if not self.per_data_buffer:
            return
        try:
            with open(self.per_data_path, 'a') as f:
                for record in self.per_data_buffer:
                    f.write(json.dumps(record) + '\n')
        except OSError as e:
            print(f"WARNING: Failed to write to PER data file: {str(e)}. Continuing without logging to file.")
        self.per_data_buffer = []
        self.per_buffer_count = 0
    
    def _format_progress_report(self, detailed=False):
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        percent_complete = (self.current_episode / config.TRAINING_EPISODES) * 100
        report = ["=================================================================="]
        report.append(f"Training Progress - Episode {self.current_episode}/{config.TRAINING_EPISODES} ({percent_complete:.1f}%)")
        report.append(f"Total Steps: {self.total_steps}")
        if self.episode_rewards:
            avg_reward = np.mean(self.reward_window) if self.reward_window else 0
            report.append(f"Avg Reward (last 100): {avg_reward:.2f}")
        if self.epsilon_values:
            latest_epsilon = self.epsilon_values[-1]
            report.append(f"Current Epsilon: {latest_epsilon:.4f}")
        if self.episode_losses:
            avg_loss = np.mean(self.loss_window) if self.loss_window else 0
            report.append(f"Avg Loss (last 100): {avg_loss:.6f}")
        report.append(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        if self.best_eval_reward > float('-inf'):
            report.append(f"Best Evaluation Reward: {self.best_eval_reward:.2f}")
        if self.episode_rewards:
            last_rewards = [f"{r:.1f}" for r in self.episode_rewards[-5:]]
            report.append(f"Last 5 Rewards: [{', '.join(last_rewards)}]")
        if detailed:
            if self.beta_values:
                latest_beta = self.beta_values[-1][1]
                beta_progress = (latest_beta - config.BETA_START) / (1.0 - config.BETA_START) * 100
                report.append(f"Current Beta: {latest_beta:.4f} ({beta_progress:.1f}% to 1.0)")
            if self.priority_means:
                latest_priority = self.priority_means[-1][1]
                report.append(f"Current Avg Priority: {latest_priority:.6f}")
            if self.td_error_means:
                latest_td_error = self.td_error_means[-1][1]
                report.append(f"Current Avg TD Error: {latest_td_error:.6f}")
            if self.is_weight_means:
                latest_is_weight = self.is_weight_means[-1][1]
                report.append(f"Current Avg IS Weight: {latest_is_weight:.6f}")
        report.append("==================================================================")
        return "\n".join(report)
