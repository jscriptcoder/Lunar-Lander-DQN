import random
import numpy as np
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer = deque(maxlen=buffer_size)  # internal buffer (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, k=self.batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    epsilon = 1e-5
    alpha = .6
    beta = .4
    beta_inc_per_sampling = 0.001
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])
        np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Sample transition
        probs  = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        self.beta = np.min([1., self.beta + self.beta_inc_per_sampling])
        
        # Compute importance-sampling weight
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return experiences, indices, weights
    
    def update(self, indices, td_errors):
        prios = np.abs(td_errors) + self.epsilon
        for idx, prio in zip(indices, prios):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)