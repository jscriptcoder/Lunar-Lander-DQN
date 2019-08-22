import random
import numpy as np
from collections import namedtuple, deque
from sum_tree import SumTree


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    p_upper = 1.
    epsilon = .01
    alpha   = .7
    beta    = .5
    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.tree  = SumTree(buffer_size)
        
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        p = self.tree.max_p
        if not p:
            p = self.p_upper
        self.tree.store(p, e)
            
    def sample(self):
        min_p   = self.tree.min_p
        seg     = self.tree.total_p / self.batch_size
        batches = []
        weights = []
        indices = []
        a = 0
        for i in range(self.batch_size):
            b = a + seg
            v = random.uniform(a, b)
            idx, p, data = self.tree.sample(v)
            
            if data is not None:
                indices.append(idx)
                weights.append((p / min_p) ** (-self.beta))
                batches.append(data)
            
            a += seg
            self.beta = min(1., self.alpha + .01)
            
        return batches, indices, weights

    def update(self, idx, tderr):
        tderr += self.epsilon
        tderr = np.minimum(tderr, self.p_upper)
        for i in range(len(idx)):
            self.tree.update(idx[i], tderr[i] ** self.alpha)
    
    def __len__(self):
        return self.tree.n_elems


class NaivePrioritizedBuffer:
    def __init__(self, buffer_size, batch_size, seed, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
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
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self, beta=0.4):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
    
    def update(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)