import random
import torch.nn.functional as F
import numpy as np

class MemoryReplay(object):

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = []

    def addMemory(self, s0, a, r, s1, done):
        memory = (s0, a, r, s1, done)
        if len(self.memory) >= self.mem_size:
            self.memory.pop(0)
        self.memory.append(memory)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            raise Exception("Not enough samples to return.")
        s0, a, r, s1, done = zip(*random.sample(self.memory, batch_size))
        return np.asarray(s0), np.asarray(a), np.asarray(r), np.asarray(s1), np.asarray(done)