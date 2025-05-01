import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import datetime
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.logger import Logger

class Visualizer:
    def __init__(self, plot_dir: str = config.PLOT_DIR, 
                 logger_instance: Optional[Logger] = None,
                 experiment_name: Optional[str] = None,
                 data_dir: Optional[str] = None):
        self.plot_dir = plot_dir
        self.logger = logger_instance
        self.experiment_name = experiment_name
        if self.logger is None:
            if experiment_name is None:
                raise ValueError("Must provide either a logger instance or an experiment name")
            if data_dir is None:
                self.data_dir = os.path.join(config.DATA_DIR, experiment_name)
            else:
                self.data_dir = os.path.join(data_dir, experiment_name)
            self.experiment_name = experiment_name
            self.episode_data_path = os.path.join(self.data_dir, "episode_data.jsonl")
            self.per_data_path = os.path.join(self.data_dir, "per_data.jsonl")
        else:
            self.experiment_name = logger_instance.experiment_name
            self.data_dir = logger_instance.data_dir
        self.plots_dir = os.path.join(plot_dir, self.experiment_name)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.epsilon_values = []
        self.beta_values = []
        self.priority_means = []
        self.priority_maxes = []
        self.td_error_means = []
        self.is_weight_means = []
        if self.logger is None:
            self._load_data_from_files()
    
    def _load_data_from_files(self):
        if os.path.exists(self.episode_data_path):
            try:
                rewards = []
                lengths = []
                losses = []
                epsilon = []
                with open(self.episode_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            rewards.append(data.get('reward', 0))
                            lengths.append(data.get('steps', 0))
                            if 'loss' in data:
                                losses.append(data.get('loss', 0))
                            if 'epsilon' in data:
                                epsilon.append(data.get('epsilon', 0))
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                self.episode_losses = losses
                self.epsilon_values = epsilon
            except Exception as e:
                print(f"Error loading episode data: {e}")
        if os.path.exists(self.per_data_path):
            try:
                beta_values = []
                priority_means = []
                priority_maxes = []
                td_error_means = []
                is_weight_means = []
                with open(self.per_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            step = data.get('step', 0)
                            if 'beta' in data:
                                beta_values.append((step, data['beta']))
                            if 'mean_priority' in data:
                                priority_means.append((step, data['mean_priority']))
                            elif 'priority_mean' in data:
                                priority_means.append((step, data['priority_mean']))
                            if 'max_priority' in data:
                                priority_maxes.append((step, data['max_priority']))
                            elif 'priority_max' in data:
                                priority_maxes.append((step, data['priority_max']))
                            if 'mean_td_error' in data:
                                td_error_means.append((step, data['mean_td_error']))
                            elif 'td_error_mean' in data:
                                td_error_means.append((step, data['td_error_mean']))
                            if 'mean_is_weight' in data:
                                is_weight_means.append((step, data['mean_is_weight']))
                            elif 'is_weight_mean' in data:
                                is_weight_means.append((step, data['is_weight_mean']))
                self.beta_values = sorted(beta_values, key=lambda x: x[0])
                self.priority_means = sorted(priority_means, key=lambda x: x[0])
                self.priority_maxes = sorted(priority_maxes, key=lambda x: x[0])
                self.td_error_means = sorted(td_error_means, key=lambda x: x[0])
                self.is_weight_means = sorted(is_weight_means, key=lambda x: x[0])
            except Exception as e:
                print(f"Error loading PER data: {e}")
    
    def _get_data(self):
        if self.logger is not None:
            data = self.logger.get_training_data()
            self.episode_rewards = data.get('rewards', [])
            self.episode_lengths = data.get('lengths', [])
            self.episode_losses = data.get('losses', [])
            self.epsilon_values = data.get('epsilon_values', [])
            self.beta_values = data.get('beta_values', [])
            self.priority_means = data.get('priority_means', [])
            self.priority_maxes = data.get('priority_maxes', [])
            self.td_error_means = data.get('td_error_means', [])
            self.is_weight_means = data.get('is_weight_means', [])
    
    def setup_plot_style(self):
        plt.rcdefaults()
        self.colors = {
            'reward_raw': '#79b6e3',
            'reward_avg': '#0066cc',
            'loss_raw': '#f48fb1',
            'loss_avg': '#cc0000',
            'epsilon': '#2ecc71',
            'beta': '#8e44ad',
            'td_error': '#e67e22',
            'priority_mean': '#00acc1',
            'priority_max': '#0d47a1',
            'background': '#f8f9fa'
        }
        self.font_sizes = {
            'title': 18,
            'subtitle': 15,
            'axis_label': 13,
            'tick_label': 11,
            'legend': 11,
            'stats': 10
        }
        self.fig_sizes = {
            'single': (14, 8),
            'overview': (16, 14),
            'multi': (14, 16)
        }
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['figure.figsize'] = self.fig_sizes['single']
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.8
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['axes.titlepad'] = 12
        plt.rcParams['axes.labelpad'] = 8

    def configure_axis(self, ax, title, xlabel, ylabel, log_scale=False):
        ax.set_title(title, fontsize=self.font_sizes['subtitle'], fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.font_sizes['axis_label'])
        ax.set_ylabel(ylabel, fontsize=self.font_sizes['axis_label'])
        ax.tick_params(labelsize=self.font_sizes['tick_label'], length=5, width=1.2)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        if log_scale:
            ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    def plot_rewards(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        self._get_data()
        if not self.episode_rewards:
            print("No reward data available for plotting.")
            return ""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        episodes = range(1, len(self.episode_rewards) + 1)
        ax.plot(episodes, self.episode_rewards, alpha=0.4, color=self.colors['reward_raw'], 
                linewidth=1.2, label='Episode Reward')
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_rewards) + 1), 
                   moving_avg, 
                   color=self.colors['reward_avg'], 
                   linewidth=2.5, 
                   label=f'{window_size}-Episode Avg')
        self.configure_axis(ax, 'Training Rewards', 'Episode', 'Reward')
        if self.episode_rewards:
            max_reward = max(self.episode_rewards)
            recent_avg = np.mean(self.episode_rewards[-min(100, len(self.episode_rewards)):])
            stats_text = f"Max: {max_reward:.2f}, Recent Avg: {recent_avg:.2f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='lower right')
        fig.suptitle(f'Training Rewards - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rewards_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""

    def plot_losses(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        self._get_data()
        if not self.episode_losses:
            print("No loss data available for plotting.")
            return ""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        episodes = range(1, len(self.episode_losses) + 1)
        ax.plot(episodes, self.episode_losses, alpha=0.4, color=self.colors['loss_raw'], 
                linewidth=1.2, label='Episode Loss')
        if len(self.episode_losses) >= window_size:
            moving_avg = np.convolve(self.episode_losses, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_losses) + 1), 
                   moving_avg, 
                   color=self.colors['loss_avg'], 
                   linewidth=2.5, 
                   label=f'{window_size}-Episode Avg')
        self.configure_axis(ax, 'Training Losses', 'Episode', 'Loss', log_scale=True)
        if self.episode_losses:
            min_loss = min(self.episode_losses)
            recent_avg = np.mean(self.episode_losses[-min(100, len(self.episode_losses)):])
            stats_text = f"Min: {min_loss:.6f}, Recent Avg: {recent_avg:.6f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='upper right')
        fig.suptitle(f'Training Losses - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"losses_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""

    def plot_epsilon(self, save: bool = True, show: bool = False) -> str:
        self._get_data()
        if not self.epsilon_values:
            print("No epsilon data available for plotting.")
            return ""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        episodes = range(1, len(self.epsilon_values) + 1)
        ax.plot(episodes, self.epsilon_values, color=self.colors['epsilon'], 
                linewidth=2, label='Epsilon')
        self.configure_axis(ax, 'Exploration Rate (Epsilon)', 'Episode', 'Epsilon Value')
        if self.epsilon_values:
            current_epsilon = self.epsilon_values[-1] if self.epsilon_values else 0
            initial_epsilon = self.epsilon_values[0] if self.epsilon_values else 0
            stats_text = f"Current: {current_epsilon:.4f}, Initial: {initial_epsilon:.4f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        fig.suptitle(f'Exploration Rate (Epsilon) - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"epsilon_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""

    def plot_per_metrics(self, save: bool = True, show: bool = False) -> str:
        self._get_data()
        if not self.beta_values and not self.priority_means:
            print("No PER metrics data available for plotting.")
            return ""
        self.setup_plot_style()
        fig, axs = plt.subplots(3, 1, figsize=self.fig_sizes['multi'])
        if self.beta_values:
            steps, betas = zip(*self.beta_values)
            axs[0].plot(steps, betas, color=self.colors['beta'], linewidth=2, label='Beta')
            self.configure_axis(axs[0], 'Importance Sampling Weight (Beta)', '', 'Beta Value')
            current_beta = betas[-1] if betas else 0
            initial_beta = betas[0] if betas else 0
            stats_text = f"Current: {current_beta:.4f}, Initial: {initial_beta:.4f}"
            axs[0].text(0.02, 0.95, stats_text, transform=axs[0].transAxes, 
                       verticalalignment='top', fontsize=self.font_sizes['stats'],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
            axs[0].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[0], 'Importance Sampling Weight (Beta) - No Data', '', '')
            axs[0].text(0.5, 0.5, "No Beta data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[0].transAxes, fontsize=12)
        if self.priority_means:
            steps_p, priorities = zip(*self.priority_means)
            axs[1].plot(steps_p, priorities, color=self.colors['priority_mean'], 
                       linewidth=2, label='Mean Priority')
            if self.priority_maxes:
                steps_m, max_priorities = zip(*self.priority_maxes)
                axs[1].plot(steps_m, max_priorities, color=self.colors['priority_max'], 
                           linewidth=1.5, alpha=0.7, label='Max Priority')
            self.configure_axis(axs[1], 'Priority Distribution', '', 'Priority Value', log_scale=True)
            axs[1].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[1], 'Priority Distribution - No Data', '', '')
            axs[1].text(0.5, 0.5, "No Priority data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[1].transAxes, fontsize=12)
        if self.td_error_means:
            steps_td, td_errors = zip(*self.td_error_means)
            axs[2].plot(steps_td, td_errors, color=self.colors['td_error'], linewidth=2, label='Mean |TD Error|')
            self.configure_axis(axs[2], 'Temporal Difference Error', 'Training Steps', 'TD Error', log_scale=True)
            axs[2].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[2], 'Temporal Difference Error - No Data', 'Training Steps', '')
            axs[2].text(0.5, 0.5, "No TD error data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[2].transAxes, fontsize=12)
        for ax in axs:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
        fig.suptitle(f'Prioritized Experience Replay Metrics - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"per_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""
    
    def plot_training_overview(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        self._get_data()
        if not (self.episode_rewards or self.episode_losses or self.epsilon_values or self.beta_values):
            print("No training metrics data available for overview plotting.")
            return ""
        self.setup_plot_style()
        fig, axs = plt.subplots(2, 2, figsize=self.fig_sizes['overview'], sharex=False)
        fig.subplots_adjust(hspace=0.25, wspace=0.2)
        axs = axs.flatten()
        metrics_plotted = 0
        for i, metric_name in enumerate(config.LOGGER_MAJOR_METRICS):
            if i >= 4:
                break
            if metric_name == "reward" and self.episode_rewards:
                episodes = range(1, len(self.episode_rewards) + 1)
                axs[metrics_plotted].plot(episodes, self.episode_rewards, alpha=0.4, color=self.colors['reward_raw'], 
                          linewidth=1.2, label='Episode Reward')
                if len(self.episode_rewards) >= window_size:
                    moving_avg = np.convolve(self.episode_rewards, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
                    axs[metrics_plotted].plot(range(window_size, len(self.episode_rewards) + 1), 
                              moving_avg, 
                              color=self.colors['reward_avg'], 
                              linewidth=2.5, 
                              label=f'{window_size}-Ep Avg')
                self.configure_axis(axs[metrics_plotted], 'Training Rewards', 'Episode', 'Reward')
                if self.episode_rewards:
                    max_reward = max(self.episode_rewards)
                    recent_avg = np.mean(self.episode_rewards[-min(100, len(self.episode_rewards)):])
                    stats_text = f"Max: {max_reward:.2f}, Recent Avg: {recent_avg:.2f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                metrics_plotted += 1
            elif metric_name == "loss" and self.episode_losses:
                episodes = range(1, len(self.episode_losses) + 1)
                axs[metrics_plotted].plot(episodes, self.episode_losses, alpha=0.4, color=self.colors['loss_raw'], 
                          linewidth=1.2, label='Episode Loss')
                if len(self.episode_losses) >= window_size:
                    moving_avg = np.convolve(self.episode_losses, np.ones(window_size)/window_size, mode='valid')
                    axs[metrics_plotted].plot(range(window_size, len(self.episode_losses) + 1), 
                              moving_avg, color=self.colors['loss_avg'], 
                              linewidth=2.5, label=f'{window_size}-Ep Avg')
                self.configure_axis(axs[metrics_plotted], 'Training Losses', 'Episode', 'Loss', log_scale=True)
                if self.episode_losses:
                    min_loss = min(self.episode_losses)
                    recent_avg = np.mean(self.episode_losses[-min(100, len(self.episode_losses)):])
                    stats_text = f"Min: {min_loss:.6f}, Recent Avg: {recent_avg:.6f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                metrics_plotted += 1
            elif metric_name == "epsilon" and self.epsilon_values:
                episodes = range(1, len(self.epsilon_values) + 1)
                axs[metrics_plotted].plot(episodes, self.epsilon_values, color=self.colors['epsilon'], 
                          linewidth=2, label='Epsilon')
                self.configure_axis(axs[metrics_plotted], 'Exploration Rate (Epsilon)', 'Episode', 'Epsilon Value')
                if self.epsilon_values:
                    current_epsilon = self.epsilon_values[-1] if self.epsilon_values else 0
                    initial_epsilon = self.epsilon_values[0] if self.epsilon_values else 0
                    stats_text = f"Initial: {initial_epsilon:.4f}, Current: {current_epsilon:.4f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                metrics_plotted += 1
            elif metric_name == "beta" and self.beta_values:
                steps, beta_values = zip(*self.beta_values)
                if self.episode_lengths:
                    cumulative_steps = np.cumsum(self.episode_lengths)
                    episode_numbers = []
                    for step in steps:
                        episode_idx = np.searchsorted(cumulative_steps, step)
                        episode_numbers.append(episode_idx + 1)
                else:
                    if self.episode_rewards:
                        total_episodes = len(self.episode_rewards)
                        max_step = max(steps) if steps else 0
                        episode_numbers = [int((step / max_step) * total_episodes) + 1 for step in steps]
                    else:
                        episode_numbers = steps
                valid_indices = [i for i, ep in enumerate(episode_numbers) if ep <= len(self.episode_rewards)]
                if valid_indices:
                    valid_episodes = [episode_numbers[i] for i in valid_indices]
                    valid_betas = [beta_values[i] for i in valid_indices]
                    axs[metrics_plotted].plot(valid_episodes, valid_betas, 
                                              color=self.colors['beta'], linewidth=2, label='Beta')
                    self.configure_axis(axs[metrics_plotted], 'Importance Sampling Weight (Beta)', 
                                        'Episode', 'Beta Value')
                    if beta_values:
                        initial_beta = beta_values[0] if beta_values else 0
                        current_beta = beta_values[-1] if beta_values else 0
                        stats_text = f"Initial: {initial_beta:.4f}, Current: {current_beta:.4f}"
                        axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                                  verticalalignment='top', fontsize=self.font_sizes['stats'],
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                    handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                    if handles and labels:
                        axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                else:
                    self.configure_axis(axs[metrics_plotted], 'Importance Sampling Weight (Beta)', 
                                        'Episode', 'Beta Value')
                    axs[metrics_plotted].text(0.5, 0.5, "Cannot map steps to episodes", 
                                            horizontalalignment='center', verticalalignment='center',
                                            transform=axs[metrics_plotted].transAxes, fontsize=12)
                metrics_plotted += 1
        for i in range(metrics_plotted, 4):
            axs[i].text(0.5, 0.5, "No data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[i].transAxes, fontsize=12)
            self.configure_axis(axs[i], "Empty Plot", "", "")
        fig.suptitle(f'Major Training Metrics - {self.experiment_name}', 
                     fontsize=self.font_sizes['title'], fontweight='bold', y=0.98)
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"major_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""

    def generate_training_config_markdown(self, save=True, show=False) -> str:
        import sys
        import os
        import inspect
        import config
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_{timestamp}.md"
        filepath = os.path.join(self.plots_dir, filename)
        config_values = {name: value for name, value in vars(config).items() 
                        if not name.startswith('__') and not inspect.ismodule(value)}
        categories = {
            "Game Environment Settings": [
                "ENV_NAME", "ACTION_SPACE_SIZE", "DIFFICULTY", 
                "FRAME_WIDTH", "FRAME_HEIGHT", "FRAME_STACK", "FRAME_SKIP", "NOOP_MAX",
                "RENDER_MODE", "TRAINING_MODE"
            ],
            "Deep Q-Learning Parameters": [
                "LEARNING_RATE", "GAMMA", "BATCH_SIZE", "MEMORY_CAPACITY",
                "TARGET_UPDATE_FREQUENCY", "TRAINING_EPISODES", "EPSILON_START",
                "EPSILON_END", "EPSILON_DECAY", "DEFAULT_EVALUATE_MODE",
                "LEARNING_STARTS", "UPDATE_FREQUENCY"
            ],
            "Prioritized Experience Replay Parameters": [
                "USE_PER", "ALPHA", "BETA_START", "BETA_FRAMES", "EPSILON_PER",
                "TREE_CAPACITY", "DEFAULT_NEW_PRIORITY", "PER_LOG_FREQUENCY", "PER_BATCH_SIZE"
            ],
            "Neural Network Settings": [
                "USE_ONE_CONV_LAYER", "USE_TWO_CONV_LAYERS", "USE_THREE_CONV_LAYERS",
                "CONV1_CHANNELS", "CONV1_KERNEL_SIZE", "CONV1_STRIDE",
                "CONV2_CHANNELS", "CONV2_KERNEL_SIZE", "CONV2_STRIDE",
                "CONV3_CHANNELS", "CONV3_KERNEL_SIZE", "CONV3_STRIDE",
                "FC_SIZE", "GRAD_CLIP_NORM"
            ],
            "Evaluation Settings": [
                "EVAL_EPISODES", "EVAL_FREQUENCY"
            ],
            "Logger Settings": [
                "MEMORY_THRESHOLD_PERCENT", "RESULTS_DIR", "LOG_DIR", "MODEL_DIR", "PLOT_DIR", "DATA_DIR",
                "ENABLE_FILE_LOGGING", "LOGGER_SAVE_INTERVAL", "LOGGER_MEMORY_WINDOW",
                "LOGGER_BATCH_SIZE", "LOGGER_DETAILED_INTERVAL", "LOGGER_MAJOR_METRICS",
                "VISUALIZATION_SAVE_INTERVAL"
            ]
        }
        markdown = f"# Training Configuration - {self.experiment_name}\n\n"
        markdown += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for category, variables in categories.items():
            markdown += f"## {category}\n\n"
            markdown += "| Parameter | Value | Description | Impact of Changes |\n"
            markdown += "| --- | --- | --- | --- |\n"
            for var in variables:
                if var in config_values:
                    value = config_values[var]
                    var_comments = ""
                    impact_comments = ""
                    try:
                        with open(inspect.getfile(config), 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for i, line in enumerate(lines):
                                if f"{var} = " in line or f"{var}=" in line:
                                    comment_parts = line.split('#')
                                    if len(comment_parts) > 1:
                                        comment_text = comment_parts[1].strip()
                                        if " - " in comment_text:
                                            parts = comment_text.split(" - ", 1)
                                            var_comments = parts[0].strip()
                                            impact_comments = parts[1].strip()
                                        else:
                                            var_comments = comment_text
                                    elif i > 0 and '#' in lines[i-1] and not '=' in lines[i-1]:
                                        var_comments = lines[i-1].split('#')[1].strip()
                    except Exception as e:
                        pass
                    if isinstance(value, str):
                        formatted_value = f"'{value}'"
                    elif isinstance(value, bool):
                        formatted_value = str(value)
                    elif isinstance(value, (int, float)):
                        formatted_value = str(value)
                    elif isinstance(value, list):
                        if len(value) > 5:
                            formatted_value = f"[{', '.join(str(v) for v in value[:5])}...]"
                        else:
                            formatted_value = str(value)
                    else:
                        formatted_value = str(value)
                    markdown += f"| {var} | {formatted_value} | {var_comments} | {impact_comments} |\n"
            markdown += "\n"
        try:
            from src.device_utils import get_system_info, get_device
            system_info = get_system_info()
            markdown += "## System Information\n\n"
            markdown += "| Component | Details |\n"
            markdown += "| --- | --- |\n"
            for key, value in system_info.items():
                markdown += f"| {key.replace('_', ' ').title()} | {value} |\n"
            device = get_device()
            markdown += f"| **Training Device** | {device} |\n"
            import torch
            if torch.cuda.is_available():
                markdown += f"| **CUDA Version** | {torch.version.cuda} |\n"
                markdown += f"| **GPU Count** | {torch.cuda.device_count()} |\n"
                for i in range(torch.cuda.device_count()):
                    markdown += f"| **GPU {i} Name** | {torch.cuda.get_device_name(i)} |\n"
                    markdown += f"| **GPU {i} Memory** | {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB |\n"
            markdown += "\n"
        except Exception as e:
            markdown += f"## System Information\n\n"
            markdown += f"*Error retrieving system information: {str(e)}*\n\n"
        import torch
        markdown += "## Runtime Information\n\n"
        markdown += "| Component | Details |\n"
        markdown += "| --- | --- |\n"
        markdown += f"| PyTorch Version | {torch.__version__} |\n"
        import sys
        markdown += f"| Python Version | {sys.version.split()[0]} |\n"
        if torch.cuda.is_available():
            markdown += f"| CUDA Available | Yes |\n"
            try:
                markdown += f"| CUDA Version | {torch.version.cuda} |\n"
            except:
                markdown += f"| CUDA Version | Unknown |\n"
        else:
            markdown += f"| CUDA Available | No |\n"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            markdown += f"| MPS (Apple Metal) Available | Yes |\n"
        else:
            markdown += f"| MPS (Apple Metal) Available | No |\n"
        markdown += "\n"
        markdown += "## Hyperparameter Tuning Suggestions\n\n"
        markdown += "Based on current settings, consider adjusting these parameters if facing issues:\n\n"
        markdown += "1. **Learning Stability Issues**:\n"
        markdown += "   - Decrease `LEARNING_RATE` to 0.0001\n"
        markdown += "   - Increase `BATCH_SIZE` to 128\n"
        markdown += "   - Decrease `TARGET_UPDATE_FREQUENCY` to 4000\n\n"
        markdown += "2. **Exploration Issues**:\n"
        markdown += "   - Increase `EPSILON_END` to 0.15-0.2\n"
        markdown += "   - Increase `EPSILON_DECAY` to slow down exploration decay\n\n"
        markdown += "3. **PER Performance Issues**:\n"
        markdown += "   - Adjust `ALPHA` between 0.4-0.8 to control prioritization strength\n"
        markdown += "   - Increase `BETA_FRAMES` to slow down bias correction\n\n"
        markdown += "4. **Memory Issues**:\n"
        markdown += "   - Decrease `MEMORY_CAPACITY` if experiencing RAM limitations\n"
        markdown += "   - Adjust `FRAME_WIDTH` and `FRAME_HEIGHT` for smaller state representations\n\n"
        if save:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                print(f"Configuration saved to {filepath}")
            except Exception as e:
                print(f"Error saving configuration markdown: {str(e)}")
                return ""
        if show:
            print(markdown)
        return filepath if save else ""

    def generate_all_plots(self, show: bool = False) -> List[str]:
        plot_files = []
        try:
            print("Generating reward plot...")
            reward_plot = self.plot_rewards(save=True, show=show)
            if reward_plot:
                plot_files.append(reward_plot)
            print("Generating loss plot...")
            loss_plot = self.plot_losses(save=True, show=show)
            if loss_plot:
                plot_files.append(loss_plot)
            try:
                print("Generating epsilon plot...")
                epsilon_plot = self.plot_epsilon(save=True, show=show)
                if epsilon_plot:
                    plot_files.append(epsilon_plot)
            except Exception as e:
                print(f"Error generating epsilon plot: {str(e)}")
            try:
                print("Generating PER metrics plot...")
                per_plot = self.plot_per_metrics(save=True, show=show)
                if per_plot:
                    plot_files.append(per_plot)
            except Exception as e:
                print(f"Error generating PER metrics plot: {str(e)}")
            try:
                print("Generating overview plot...")
                major_metrics_plot = self.plot_training_overview(save=True, show=show)
                if major_metrics_plot:
                    plot_files.append(major_metrics_plot)
            except Exception as e:
                print(f"Error generating overview plot: {str(e)}")
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            import traceback
            traceback.print_exc()
        if not plot_files:
            print("No plots were generated. This might be due to missing training data.")
        else:
            print(f"Successfully generated {len(plot_files)} plots.")
        return plot_files

    def avg_reward(self, window_size: int = 100, save: bool = True, show: bool = True) -> float:
        self._get_data()
        if not self.episode_rewards:
            print("No reward data available for plotting.")
            return ""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        total_episodes = len(self.episode_rewards)
        num_windows = total_episodes // window_size
        avg_rewards = []
        x_values = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            if end_idx <= total_episodes:
                avg_rewards.append(np.mean(self.episode_rewards[start_idx:end_idx]))
                x_values.append(end_idx)
        ax.plot(x_values, avg_rewards, color=self.colors['reward_avg'], 
            linewidth=2.5, label=f'Average Reward (per {window_size} episodes)')
        self.configure_axis(ax, 'Training Rewards (Averaged)', 'Episode', 'Average Reward')
        if avg_rewards:
            max_avg_reward = max(avg_rewards)
            recent_avg = avg_rewards[-1]
            stats_text = f"Max Avg: {max_avg_reward:.2f}, Recent Avg: {recent_avg:.2f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=self.font_sizes['stats'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='lower right')
        fig.suptitle(f'Training Rewards (Averaged) - {self.experiment_name}', 
                fontsize=self.font_sizes['title'], fontweight='bold')
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rewards_avg_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath if save else ""

if __name__ == "__main__":
    results_dir = config.RESULTS_DIR
    data_dir = config.DATA_DIR
    specific_experiment = f'exp_{config.VISUALIZATION_SPECIFIC_EXPERIMENT}'
    print(f"Looking for experiments in {data_dir}")
    if os.path.exists(data_dir):
        experiments = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        if experiments:
            if specific_experiment:
                latest_experiment = specific_experiment
                print(f"Found {len(experiments)} experiments. Using the most recent: {latest_experiment}")
            else:
                experiments.sort()
                latest_experiment = experiments[-1]
                print(f"Found {len(experiments)} experiments. Using the most recent: {latest_experiment}")
            vis = Visualizer(experiment_name=latest_experiment)
            vis.avg_reward()
            print(f"Generating plots from experiment: {latest_experiment}")
            plot_files = vis.generate_all_plots(show=True)
            print(f"Generated {len(plot_files)} plots: {plot_files}")
        else:
            print("No experiments found. Run training first to generate data.")
    else:
        print(f"Data directory {data_dir} not found. Please run training first to generate data.")
