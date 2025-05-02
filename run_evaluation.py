import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from preprocessing import AtariFrameProcessor
from models import DuelingQNetwork
from utils import set_random_seeds
from wrappers import FrameSkipWrapper, EpisodicLifeWrapper, FireResetWrapper


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Dueling DQN agent'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the trained model checkpoint'
    )
    
    parser.add_argument(
        '--env', 
        type=str, 
        default='ALE/MsPacman-v5',
        help='Gymnasium environment name'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10,
        help='Number of episodes to evaluate'
    )
    
    parser.add_argument(
        '--render', 
        action='store_true',
        help='Render the environment'
    )
    
    parser.add_argument(
        '--record', 
        action='store_true',
        help='Record videos of evaluation episodes'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def make_atari_environment(env_name, render_mode=None):
   
    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Check if the environment is actually an Atari game
    if 'ALE/' in env_name:
        # Apply frame skipping (4 frames)
        env = FrameSkipWrapper(env, skip=4, max_pool=True)
        
        # Apply episodic life (terminal on life loss)
        env = EpisodicLifeWrapper(env)
        
        # Apply fire reset if applicable (games that need "FIRE" to start)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
    
    return env


def evaluate_model(model_path, env_name, num_episodes, render=False, record=False, seed=42):
   
    # Set random seeds
    set_random_seeds(seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create environment
    render_mode = 'human' if render else None
    env = make_atari_environment(env_name, render_mode=render_mode)
    
    # Setup video recording if requested
    if record:
        videos_dir = os.path.join(os.path.dirname(model_path), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=videos_dir,
            episode_trigger=lambda episode_id: True,  # Record all episodes
            name_prefix="evaluation"
        )
    
    # Get environment details
    action_count = env.action_space.n
    
    # Initialize frame processor
    frame_processor = AtariFrameProcessor()
    
    # Create and load model
    model = DuelingQNetwork(4, action_count).to(device)
    model.load_state_dict(checkpoint['policy_network'])
    model.eval()
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = frame_processor.initialize(obs)
        
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action using greedy policy
            state_tensor = torch.FloatTensor(np.array([state])).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Process new frame
            state = frame_processor.update(obs)
            
            # Update metrics
            episode_reward += reward
            steps += 1
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}/{num_episodes} - "
              f"Reward: {episode_reward} - "
              f"Steps: {steps}")
    
    # Close environment
    env.close()
    
    # Calculate summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {mean_length:.2f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_episodes), episode_rewards)
    plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Evaluation Results - {env_name}')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(model_path), 'evaluation_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length
    }


def main():
    args = parse_arguments()
    
    print(f"Evaluating model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print(f"Recording: {'Enabled' if args.record else 'Disabled'}")
    
    evaluate_model(
        model_path=args.model,
        env_name=args.env,
        num_episodes=args.episodes,
        render=args.render,
        record=args.record,
        seed=args.seed
    )


if __name__ == "__main__":
    main()