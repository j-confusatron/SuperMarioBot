import random
import torch
import numpy as np

class MemoryReplay(object):

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = [None for _ in range(mem_size)]
        self.i = 0
        self.filled = 0

    def addMemory(self, s0, a, r, s1, done):
        if self.i >= self.mem_size:
            self.i = 0
        elif self.filled < self.mem_size:
            self.filled += 1
        self.memory[self.i] = (s0, a, r, s1, done)
        self.i += 1

    def sample(self, batch_size, device):
        if self.filled < batch_size:
            raise Exception("Not enough samples to return.")
        if self.filled < self.mem_size:
            s0, a, r, s1, done = zip(*random.sample(self.memory[:self.filled], batch_size))
            s0, a, r, s1, done = map(torch.tensor, [np.stack(s0), a, r, np.stack(s1), done])
        else:
            s0, a, r, s1, done = zip(*random.sample(self.memory, batch_size))
            s0, a, r, s1, done = map(torch.tensor, [np.stack(s0), a, r, np.stack(s1), done])
        return s0.to(device), a.to(device), r.to(device), s1.to(device), done.to(device)