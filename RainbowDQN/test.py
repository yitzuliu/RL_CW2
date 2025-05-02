import os
import matplotlib.pyplot as plt
import torch

from environment import Environment



def test(args, steps, dqn, val_mem, log_data, output_dir, evaluate=False):
  env = Environment(args)
  env.eval()
  if steps > 0:  
    log_data['steps'].append(steps)
  raw_rewards, shaped_rewards, tot_Qs = [], [], []

  for episode in range(args.evaluation_episodes):
    state = env.reset()
    done = False
    raw_reward_sum = 0
    shaped_reward_sum = 0
    step_count = 0
    max_steps = 10000 
    
    print(f"Starting evaluation episode {episode+1}/{args.evaluation_episodes}")
    
    while not done and step_count < max_steps:
      step_count += 1
      action = dqn.act_e_greedy(state) 
      state, raw_reward, shaped_reward, done = env.step(action)
      raw_reward_sum += raw_reward
      shaped_reward_sum += shaped_reward
      
      if step_count % 100 == 0:
        print(f"  Step {step_count}, Raw: {raw_reward_sum:.1f}, Shaped: {shaped_reward_sum:.1f}")
      
      if args.render:
        env.render()
    
    print(f"Episode {episode+1} finished: Steps={step_count}, Raw={raw_reward_sum:.1f}, Shaped={shaped_reward_sum:.1f}")
    raw_rewards.append(raw_reward_sum)
    shaped_rewards.append(shaped_reward_sum)

  env.close()

  for state in val_mem:
    tot_Qs.append(dqn.evaluate_q(state))

  avg_raw_reward = sum(raw_rewards) / len(raw_rewards)
  avg_shaped_reward = sum(shaped_rewards) / len(shaped_rewards)
  avg_Q = sum(tot_Qs) / len(tot_Qs)
  
  print(f"Evaluation results (detailed):")
  print(f"  Raw rewards per episode: {[round(r, 1) for r in raw_rewards]}")
  print(f"  Shaped rewards per episode: {[round(r, 1) for r in shaped_rewards]}")
  print(f"  Average raw: {avg_raw_reward:.2f}, Average shaped: {avg_shaped_reward:.2f}")
  
  if not evaluate:
    if avg_shaped_reward > log_data['best_avg_reward']:
      log_data['best_avg_reward'] = avg_shaped_reward
      dqn.save(output_dir)

    log_data['raw_rewards'].append(raw_rewards)
    log_data['shaped_rewards'].append(shaped_rewards)
    log_data['Qs'].append(tot_Qs)
    torch.save(log_data, os.path.join(output_dir, 'log_data.pth'))

    plot_line(log_data['steps'], log_data['raw_rewards'], 'Raw Reward', path=output_dir)
    plot_line(log_data['steps'], log_data['shaped_rewards'], 'Shaped Reward', path=output_dir)
    plot_line(log_data['steps'], log_data['Qs'], 'Q', path=output_dir)
    
    compare_plot(log_data['steps'], log_data['raw_rewards'], log_data['shaped_rewards'], path=output_dir)
    
    if len(log_data['learning_rates']) > 0:
      _plot_learning_rate(log_data['learning_rates'], path=output_dir)
      
    if len(log_data['losses']) > 0:
      _plot_loss(log_data['losses'], path=output_dir)

  return avg_shaped_reward, avg_Q, avg_raw_reward


def plot_line(xs, population, title, path=''):
  plt.figure(figsize=(10, 5))
  
  data = torch.tensor(population, dtype=torch.float32)
  data_min, data_max = data.min(1)[0].squeeze().numpy(), data.max(1)[0].squeeze().numpy()
  data_mean,data_std = data.mean(1).squeeze().numpy(), data.std(1).squeeze().numpy()
  data_upper, data_lower = data_mean + data_std, data_mean - data_std
  
  plt.plot(xs, data_max, '--', color='blue', alpha=0.5, label='Max')
  plt.plot(xs, data_min, '--', color='blue', alpha=0.5, label='Min')
  plt.plot(xs, data_mean, '-', color='blue', label='Mean')
  plt.fill_between(xs, data_lower, data_upper, color='blue', alpha=0.2, label='±1 Std. Dev.')
  
  plt.title(title)
  plt.xlabel('Step')
  plt.ylabel(title)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(path, f"{title.replace(' ', '_')}.png"))
  plt.close()

def compare_plot(xs, raw_ys_population, shaped_ys_population, path=''):
  plt.figure(figsize=(10, 5))
  
  raw_ys = torch.tensor(raw_ys_population, dtype=torch.float32)
  shaped_ys = torch.tensor(shaped_ys_population, dtype=torch.float32)
  
  raw_mean = raw_ys.mean(1).squeeze().numpy()
  shaped_mean = shaped_ys.mean(1).squeeze().numpy()
  
  plt.plot(xs, raw_mean, '-', color='blue', label='Raw Rewards')
  plt.plot(xs, shaped_mean, '-', color='red', label='Shaped Rewards')
  
  plt.title('Raw vs Shaped Rewards')
  plt.xlabel('Step')
  plt.ylabel('Mean Reward')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(path, "Reward_Comparison.png"))
  plt.close()

def _plot_learning_rate(lr_data, path=''):
  plt.figure(figsize=(10, 5))
  
  steps = [x[0] for x in lr_data]
  lrs = [x[1] for x in lr_data]
  
  plt.plot(steps, lrs, '-o', color='green')
  plt.title('Learning Rate Schedule')
  plt.xlabel('Step')
  plt.ylabel('Learning Rate')
  plt.yscale('log')  
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(os.path.join(path, "Learning_Rate.png"))
  plt.close()

def _plot_loss(loss_data, path=''):
  plt.figure(figsize=(10, 5))
  steps = [x[0] for x in loss_data]
  losses = [x[1] for x in loss_data]
  plt.plot(steps, losses, '-', color='purple')
  plt.title('Training Loss')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(os.path.join(path, "Training_Loss.png"))
  plt.close()
