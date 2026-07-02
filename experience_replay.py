import random
from collections import deque


class ReplayMemory:
    """
    Circular experience replay buffer.
    Stores (state, action, next_state, reward, terminated) tuples.
    """
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
