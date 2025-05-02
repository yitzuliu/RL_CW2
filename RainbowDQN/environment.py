import torch
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
from collections import deque
from utils import process_frame, perform_frame_skip, reshape_reward_pacman



class Environment:
    def __init__(self, args):
        self.device = args.device
        self.window = args.history_length

        self.env = gym.make(
            "ALE/MsPacman-v5",
            render_mode="rgb_array" if args.render else None,
            obs_type="grayscale",
            frameskip=1,
        )

        self.env.action_space.seed(args.rand_seed)
        _, initial_info = self.env.reset(seed=args.rand_seed)

        self.action_space_size = self.env.action_space.n
        self.actions = {i: i for i in range(self.action_space_size)}

        self.state_buffer = deque([], maxlen=self.window)
        self._initial_reset(initial_info) 

        self.life_termination = False
        self.training = True

    def _initial_reset(self, initial_info):
        self.reset_buffer()
        observation, _ = self.env.reset()
        state = process_frame(observation, self.device)
        self.state_buffer.append(state)
        self.lives = initial_info.get("lives", 0)


    def reset_buffer(self):
        self.state_buffer.clear()
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False
            observation, _, _, _, info = self.env.step(0)
            self.lives = info.get("lives", 0)
            state = process_frame(observation, self.device)
            self.state_buffer.append(state)

        else:
            observation, info = self.env.reset()
            self.reset_buffer() 
            state = process_frame(observation, self.device)
            self.state_buffer.append(state)
            self.lives = info.get("lives", 0)

        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        initial_lives = self.lives

        processed_observation, raw_reward_sum, done, info = (
            perform_frame_skip(
            self.env, self.actions, action, process_frame, self.device
        )
        )

        self.state_buffer.append(processed_observation)

        current_lives = info.get("lives", 0)
        died = current_lives < initial_lives
        level_finished = info.get("level_completed", False) 

        if died and current_lives > 0:
            self.life_termination = not done
            done = True
        self.lives = current_lives 

        reshaped_reward = reshape_reward_pacman(
            raw_reward_sum, died, level_finished
        )

        return (
        torch.stack(list(self.state_buffer), 0),
        raw_reward_sum,
        reshaped_reward,
        done,
        )


    def action_space(self):
        return self.action_space_size

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

