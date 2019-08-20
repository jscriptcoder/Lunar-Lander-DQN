import random
import numpy as np
from collections import namedtuple, deque

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
    """Naive Prioritized Experience Replay buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done", 
                                                  "priority"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):      
        # By default set max priority level
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        
        e = self.experience(state, action, reward, next_state, done, max_priority)
        
        self.memory.append(e)
    
    def sample(self, alpha=0.6, beta=0.4):
        # Probabilities associated with each entry in memory
        priorities = np.array([sample.priority for sample in self.memory])
        probs  = priorities ** alpha
        probs /= probs.sum()
        
        # Get indices
        indices = np.random.choice(len(self.memory), self.batch_size, replace = False, p=probs)
        
        # Associated experiences
        experiences = [self.memory[idx] for idx in indices]    

        # Importance sampling weights
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        return experiences, weights, indices

    def update_priorities(self, indices, priorities):
        for i, idx in enumerate(indices):
            # A tuple is immutable so need to use "_replace" method to update it - might replace the named tuple by a dict
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])
            
    def __len__(self):
        return len(self.memory)