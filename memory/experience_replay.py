from collections import deque
import random

class Memory():
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)