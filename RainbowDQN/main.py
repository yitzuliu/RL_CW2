import os
import numpy as np
import torch
import typer
import gzip
import pickle
from tqdm import trange

from environment import Environment
from agent import Agent
from memory_replay import ReplayMemory
from test import test
from utils import log_on_terminal

app = typer.Typer(pretty_exceptions_show_locals=False)

def create_args_from_options(options):
    args = type('Args', (), {})()
    for key, value in options.items():
        setattr(args, key, value)
    return args

def setup_environment(args):
    output_dir = os.path.join('output', args.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.random.seed(args.rand_seed)
    torch.manual_seed(np.random.randint(1, 10000))
    
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
    else:
        args.device = torch.device('cpu')
    print("Device Selected: ", args.device)
    
    env = Environment(args)
    env.train()
    dqn = Agent(args, env)
    
    return env, dqn, output_dir

def load_or_create_memory(args):
    def load_memory(memory_path):
        with gzip.open(memory_path + '.gz', 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)

    if args.trained_model is not None and not args.evaluate:
        if not args.memory_path:
            raise ValueError("Cannot resume training without memory save path")
        elif not os.path.exists(args.memory_path + '.gz'):
            raise ValueError(f'Could not find memory file at {args.memory_path}.gz. Aborting...')
        mem = load_memory(args.memory_path)
        if mem is None:
            raise ValueError(f"Failed to load memory from {args.memory_path}.gz")
    else:
        mem = ReplayMemory(args, args.memory_capacity)
    
    return mem

def save_memory(memory, memory_path):
    with gzip.open(memory_path + '.gz', 'wb') as zipped_pickle_file:
        pickle.dump(memory, zipped_pickle_file)

def setup_validation_memory(args, env):
    val_mem = ReplayMemory(args, args.evaluation_size)
    time_step, done = 0, True
    
    log_on_terminal("Setting up validation memory...", 1)
    while time_step < args.evaluation_size:
        if done:
            state = env.reset()
        
        action = np.random.randint(0, env.action_space())
        next_state, raw_reward, shaped_reward, done = env.step(action)
        
        val_mem.append(state, action, shaped_reward, done)
        state = next_state
        time_step += 1
    
    return val_mem

def process_episode_end(episode_count, current_raw_reward, current_shaped_reward, 
                         episode_raw_rewards, episode_shaped_rewards):
    episode_count += 1
    episode_raw_rewards.append(current_raw_reward)
    episode_shaped_rewards.append(current_shaped_reward)
    
    avg_raw = sum(episode_raw_rewards[-100:]) / min(len(episode_raw_rewards), 100)
    avg_shaped = sum(episode_shaped_rewards[-100:]) / min(len(episode_shaped_rewards), 100)
    
    log_on_terminal(
        f'Episode {episode_count} | Raw reward: {current_raw_reward:.1f} | '
        f'Shaped reward: {current_shaped_reward:.1f} | '
        f'Avg raw(100): {avg_raw:.1f} | Avg shaped(100): {avg_shaped:.1f}'
    )
    
    return episode_count, 0, 0 

def handle_learning_rate_schedule(dqn, steps, args, log_data):
    if args.learn_shlr and steps % 500000 == 0 and steps > args.learn_start + 500000:
        dqn.reduce_learning_rate(0.5)
        current_lr = dqn.get_current_lr()
        log_on_terminal(f'Reduced learning rate to {current_lr}', 1)
        log_data['learning_rates'].append((steps, current_lr))

def perform_training_step(state, dqn, env, mem, args):
    action = dqn.act(state)  
    next_state, raw_reward, shaped_reward, done = env.step(action)
    
    if args.reward_clip > 0:
        shaped_reward = max(min(shaped_reward, args.reward_clip), -args.reward_clip)
    
    mem.append(state, action, shaped_reward, done)
    
    return next_state, raw_reward, shaped_reward, done

def update_model(dqn, mem, steps, args, log_data, total_loss, loss_count, val_mem, output_dir):
    if getattr(args, 'priority_weight_increase', 0) > 0:
        mem.priority_weight = min(mem.priority_weight + args.priority_weight_increase, 1)
    
    handle_learning_rate_schedule(dqn, steps, args, log_data)
    
    if steps % args.replay_frequency == 0:
        loss = dqn.learn(mem) 
        total_loss += loss
        loss_count += 1
        
        if steps % (args.replay_frequency * 1000) == 0:
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            log_on_terminal(f'steps: {steps:>8} | Beta: {mem.priority_weight:.4f} | Loss: {avg_loss:.4f}')
            log_data['losses'].append((steps, avg_loss))
            total_loss = 0
            loss_count = 0
    
    if steps % args.evaluation_interval == 0:
        run_evaluation(dqn, args, steps, mem, log_data, val_mem, output_dir)
    
    if steps % args.target_update == 0:
        dqn.update_target_net()
        log_on_terminal(f'Updated target network at step={steps}')
    
    if (args.checkpoint_interval != 0) and (steps % args.checkpoint_interval == 0):
        checkpoint_path = f'checkpoint_{steps//1000}k.pth'
        dqn.save(output_dir, checkpoint_path)
        log_on_terminal(f'Saved checkpoint to {checkpoint_path}', 1)
    
    return total_loss, loss_count

def run_evaluation(dqn, args, steps, mem, log_data, val_mem, output_dir):
    dqn.eval()
    avg_shaped_reward, avg_Q, avg_raw_reward = test(
        args, steps, dqn, val_mem, log_data, output_dir
    ) 
    log_on_terminal(
        f'[EVAL] steps={steps} | Raw: {avg_raw_reward:.2f} | '
        f'Shaped: {avg_shaped_reward:.2f} | Q: {avg_Q:.2f} | '
        f'Beta: {mem.priority_weight:.4f}', 1
    )
    dqn.train()
    
    if args.memory_path is not None:
        save_memory(mem, args.memory_path)

def train_agent(args, env, dqn, mem, val_mem, output_dir, log_data):
    episode_raw_rewards = []
    episode_shaped_rewards = []
    current_raw_reward = 0
    current_shaped_reward = 0
    episode_count = 0
    total_loss = 0
    loss_count = 0
    
    if args.annealing_end <= 0:
        args.priority_weight_increase = 0  
    else:
        args.priority_weight_increase = (1 - args.priority_weight) / min(args.annealing_end, args.total_steps - args.learn_start)
    
    log_on_terminal("-" * 60, 2)
    log_on_terminal(f"STARTING TRAINING SESSION WITH {args.total_steps} STEPS (~{args.total_steps*4:,} FRAMES)", 2)
    log_on_terminal(f"Learning rate: {args.learning_rate} | Target update: {args.target_update} | Replay frequency: {args.replay_frequency}", 1)
    log_on_terminal("-" * 60, 2)
    
    dqn.train()
    done = True
    state = None
    
    for steps in trange(1, args.total_steps + 1):
        if done:
            state = env.reset()
            
            if steps > 1:
                episode_count, current_raw_reward, current_shaped_reward = process_episode_end(
                    episode_count, current_raw_reward, current_shaped_reward,
                    episode_raw_rewards, episode_shaped_rewards
                )
        
        if steps % args.replay_frequency == 0:
            dqn.reset_noise() 
        
        state, raw_reward, shaped_reward, done = perform_training_step(
            state, dqn, env, mem, args
        )
        
        current_raw_reward += raw_reward
        current_shaped_reward += shaped_reward
        
        if steps >= args.learn_start:
            total_loss, loss_count = update_model(
                dqn, mem, steps, args, log_data, total_loss, loss_count, val_mem, output_dir
            )
    
    dqn.save(output_dir, 'model.pth')
    if args.memory_path is not None:
        save_memory(mem, args.memory_path)
    
    log_on_terminal("=" * 80, 2)
    log_on_terminal(f"TRAINING COMPLETED: {episode_count} EPISODES | {args.total_steps} STEPS | {args.total_steps*4:,} FRAMES", 2)
    log_on_terminal("=" * 80, 2)

def evaluate_agent(args, dqn, val_mem, output_dir, log_data):
    log_on_terminal("Starting evaluation...", 2)
    dqn.eval()
    avg_shaped_reward, avg_Q, avg_raw_reward = test(
        args, 0, dqn, val_mem, log_data, output_dir, evaluate=True
    )
    log_on_terminal(
        f'Avg. raw reward: {avg_raw_reward:.2f} | '
        f'Avg. shaped reward: {avg_shaped_reward:.2f} | '
        f'Avg. Q: {avg_Q:.2f}', 1
    )

@app.command(help="Rainbow DQN for MsPacman Atari Game")
def main(
    name: str = typer.Option("mspacman_10k_test", help='Experiment name'),
    rand_seed: int = typer.Option(123, help='Random rand_seed'),
    total_steps: int = typer.Option(10000, '--T-max', help='Number of training steps'),
    max_episode_length: int = typer.Option(int(108e3), help='Max episode length'), 
    history_length: int = typer.Option(4, help='Number of consecutive states processed'),
    hidden_size: int = typer.Option(512, help='Network hidden size'),
    noisy_std: float = typer.Option(0.1, help='Initial standard deviation of noisy linear layers'),
    atoms: int = typer.Option(51, help='Discretised size of value distribution'),
    v_min: float = typer.Option(-10.0, '--V-min', help='Minimum of value distribution support'),
    v_max: float = typer.Option(10.0, '--V-max', help='Maximum of value distribution support'),
    trained_model: str = typer.Option(None, help='Pretrained model (state dict)'),
    memory_capacity: int = typer.Option(5000, help='Experience replay memory capacity'),
    replay_frequency: int = typer.Option(4, help='Frequency of sampling from memory'),
    priority_exponent: float = typer.Option(0.5, help='Prioritised experience replay exponent'),
    priority_weight: float = typer.Option(0.4, help='Initial PER importance sampling weight'),
    multi_step: int = typer.Option(3, help='Number of steps for multi-step return'),
    discount: float = typer.Option(0.99, help='Discount factor'),
    target_update: int = typer.Option(2500, help='Steps between target network updates'), 
    reward_clip: int = typer.Option(0, help='Reward clipping (0 to disable)'), 
    learning_rate: float = typer.Option(0.000020, help='Learning rate'), 
    adam_eps: float = typer.Option(1.5e-4, help='Adam epsilon'), 
    batch_size: int = typer.Option(32, help='Batch size'), 
    norm_clip: float = typer.Option(10.0, help='Max L2 norm for gradient clipping'),
    learn_start: int = typer.Option(1000, help='Steps before starting training'), 
    evaluate: bool = typer.Option(False, help='Evaluate only'),
    evaluation_interval: int = typer.Option(5000, help='Steps between evaluations'), 
    evaluation_episodes: int = typer.Option(2, help='Number of evaluation episodes'), 
    evaluation_size: int = typer.Option(100, help='Transitions for validating Q'), 
    render: bool = typer.Option(False, help='Display screen'), 
    checkpoint_interval: int = typer.Option(0, help='How often to checkpoint (0=disable)'), 
    memory_path: str = typer.Option(None, help='Path to save/load memory'),
    annealing_end: int = typer.Option(0, help='Step for beta annealing end (0=disable)'), 
    learn_shlr: bool = typer.Option(False, help='Enable learning rate scheduling'),
):
    options = locals()
    args = create_args_from_options(options)
    
    args.V_min = v_min  
    args.V_max = v_max  
    
    print(' ' * 26 + 'Configuration')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ' : ' + str(v))
    
    log_data = {
        'steps': [], 
        'raw_rewards': [], 
        'shaped_rewards': [], 
        'Qs': [], 
        'learning_rates': [],
        'losses': [],
        'best_avg_reward': -float('inf')
    }
    
    env, dqn, output_dir = setup_environment(args)
    
    mem = load_or_create_memory(args)
    val_mem = setup_validation_memory(args, env)
    
    if not args.evaluate:
        train_agent(args, env, dqn, mem, val_mem, output_dir, log_data)
    else:
        evaluate_agent(args, dqn, val_mem, output_dir, log_data)
    
    env.close()

if __name__ == "__main__":
    app()
