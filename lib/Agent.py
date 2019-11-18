import torch
import numpy as np
from lib.ImageProcess import ImageProcessing

class Agent():
    def __init__(self, env, memory):
        self.env = env
        self.memory = memory
        self.env_state = None
        self.image_processing = ImageProcessing()
        self.env_set = False

    def reset_environtment(self):
        self.env_state = self.env.reset()
        self.env_set = True
        self.total_reward = 0.0

    def use_state(self, network, device):
        assert (self.env_set), "Please initialize game before playing by resetting or inputting a state"
        state = np.array([self.env_state], copy=False)
        state = torch.tensor(state).to(device)
        q_values = network(state)
        _, pref_action = torch.max(q_values, dim=1)
        action = pref_action.item()
        return action

    def play_action(self, network, e=0, device='cpu'):
        done_reward = None
        if np.random.uniform(0, 1) < e:
            # Taking random action
            action = self.env.action_space.sample()
        else:
            action = self.use_state(network, device)

        # Do action in the environment
        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        old_state = self.env_state
        self.env_state = new_state

        self.memory.save_to_memory(old_state, action, self.env_state, reward)

        if done:
            done_reward = self.total_reward
            self.reset_environtment()

        return done_reward

