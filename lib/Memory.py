import random
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'new_state'))
class Memory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def save_to_memory(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states)
