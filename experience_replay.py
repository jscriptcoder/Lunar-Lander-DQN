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

class PrioritizedReplayBuffer:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, buffer_size, batch_size, seed):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])
        random.seed(seed)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, state, action, reward, next_state, done, error=1.0):
        p = self._get_priority(error)
        e = self.experience(state, action, reward, next_state, done)
        self.tree.add(p, e)

    def sample(self):
        experiences = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            if data is not None:
                priorities.append(p)
                experiences.append(data)
                idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()

        return experiences, idxs, weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.n_entries