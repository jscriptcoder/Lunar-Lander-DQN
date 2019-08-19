import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork, DuelingQNetwork
from experience_replay import ReplayBuffer
from device import device

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class DQNAgent():

    def __init__(self, 
                 state_size, action_size, 
                 use_double=False, use_dueling=False, use_priority=False, 
                 seed=42):
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_priority = use_priority
        self.seed = random.seed(seed)

        # Q-Network
        if use_dueling:
            self.qn_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qn_local = QNetwork(state_size, action_size, seed).to(device)
        
        if use_dueling:
            self.qn_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qn_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
#                experiences, idxs = self.memory.sample()
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qn_local.eval()
        with torch.no_grad():
            action_values = self.qn_local(state)
        self.qn_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        
        # Get expected Q values from local model
        Q_expected = self.qn_local(states).gather(-1, actions)
        
        if self.use_double: # uses Double Deep Q-Network
            
            # Get the best action using local model
            best_action = self.qn_local(next_states).argmax(-1, keepdim=True)
            
            # Evaluate the action using target model
            max_Q = self.qn_target(next_states).detach().gather(-1, best_action)
        
        else: # normal Deep Q-Network
            
            # Get max predicted Q value (for next states) from target model
            max_Q = self.qn_target(next_states).detach().max(-1, keepdim=True)[0]
            
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * max_Q * (1 - dones))

        # Compute loss and minimize
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qn_local, self.qn_target, TAU)    
        
#        return loss.cpu().numpy()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save_weights(self, path='double_local_weights.pth'):
        torch.save(self.qn_local.state_dict(), path)
    
    def load_weights(self, path='double_local_weights.pth'):
        self.qn_local.load_state_dict(torch.load(path))