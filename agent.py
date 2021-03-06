import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork, DuelingQNetwork
from experience_replay import ReplayBuffer, PrioritizedReplayBuffer
from device import device


class DQNAgent:

    def __init__(self, 
                 state_size, 
                 action_size, 
                 buffer_size=int(1e5), 
                 batch_size=64, 
                 gamma=.99, 
                 tau=1e-3, 
                 lr=5e-4, 
                 update_every=4, 
                 use_double=False, 
                 use_dueling=False, 
                 use_priority=False, 
                 use_noise=False,
                 seed=42):
        """Deep Q-Network Agent
        
        Args:
            state_size (int)
            action_size (int)
            buffer_size (int): Experience Replay buffer size
            batch_size (int)
            gamma (float): 
                discount factor, used to balance immediate and future reward
            tau (float): interpolation parameter for soft update target network
            lr (float): neural Network learning rate, 
            update_every (int): how ofter we're gonna learn, 
            use_double (bool): whether or not to use double networks improvement
            use_dueling (bool): whether or not to use dueling network improvement
            use_priority (bool): whether or not to use priority experience replay
            use_noise (bool): whether or not to use noisy nets for exploration
            seed (int)
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.lr=lr
        self.update_every=update_every
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_priority = use_priority
        self.use_noise = use_noise
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Q-Network
        if use_dueling:
            self.qn_local = DuelingQNetwork(state_size, 
                                            action_size, 
                                            noisy=use_noise).to(device)
        else:
            self.qn_local = QNetwork(state_size, 
                                     action_size, 
                                     noisy=use_noise).to(device)
        
        if use_dueling:
            self.qn_target = DuelingQNetwork(state_size, 
                                             action_size, 
                                             noisy=use_noise).to(device)
        else:
            self.qn_target = QNetwork(state_size, 
                                      action_size, 
                                      noisy=use_noise).to(device)
        
        # Initialize target model parameters with local model parameters
        self.soft_update(1.0)
        
        # TODO: make the optimizer configurable
        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=lr)

        if use_priority:
            self.memory = PrioritizedReplayBuffer(buffer_size, batch_size)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size)
            
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Step performed by the agent 
        after interacting with the environment and receiving feedback
        
        Args:
            state (int)
            action (int)
            reward (float)
            next_state (int)
            done (bool)
        """
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                
                if self.use_priority:
                    experiences, indices, weights = self.memory.sample()
                    self.learn(experiences, indices, weights)
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, state, eps=0.):
        """Given a state what's the next action to take
        
        Args:
            state (int)
            eps (flost): 
                controls how often we explore before taking the greedy action
        
        Returns:
            int: action to take
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qn_local.eval()
        with torch.no_grad():
            action_values = self.qn_local(state)
        self.qn_local.train()

        if self.use_noise:
            return np.argmax(action_values.cpu().numpy())
        else:
            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, indices=None, weights=None):
        """Use a batch of experiences to calculate TD errors and update Q networks
        
        Args:
            experiences: tuple with state, action, reward, next_state and done
            indices (Numpy array): 
                array of indices to update priorities (only used with PER)
            weights (Numpy array): 
                importance-sampling weights (only used with PER)
        """
        
        states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None]))\
                .float().to(device)
        actions = torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None]))\
                .long().to(device)
        rewards = torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None]))\
                .float().to(device)
        next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None]))\
                .float().to(device)
        dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None])\
                .astype(np.uint8)).float().to(device)
        
        if self.use_priority:
            weights = torch.from_numpy(np.vstack(weights)).float().to(device)
        
        if self.use_double: # uses Double Deep Q-Network
            
            # Get the best action using local model
            best_action = self.qn_local(next_states).argmax(-1, keepdim=True)
            
            # Evaluate the action using target model
            max_q = self.qn_target(next_states).detach().gather(-1, best_action)
        
        else: # normal Deep Q-Network
            
            # Get max predicted Q value (for next states) from target model
            max_q = self.qn_target(next_states).detach().max(-1, keepdim=True)[0]
            
        
        # Compute Q targets for current states 
        q_targets = rewards + (self.gamma * max_q * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.qn_local(states).gather(-1, actions)

        # Compute loss...
        if self.use_priority:
            # Calculate TD error to update priorities
            weighted_td_errors = weights * (q_targets - q_expected) ** 2 
            loss  = weighted_td_errors.mean()
        else:
            loss = F.mse_loss(q_expected, q_targets)
        
        # ...and minimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        if self.use_priority:
            self.memory.update(indices, weighted_td_errors.detach().cpu().numpy())

        # Update target network
        self.soft_update(self.tau)    

    def soft_update(self, tau):
        """Soft update model parameters:
            θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(self.qn_target.parameters(), 
                                             self.qn_local.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)
    
    def make_filename(self, filename):
        filename = 'noisy_' + filename if self.use_noise else filename
        filename = 'dueling_' + filename if self.use_dueling else filename
        filename = 'double_' + filename if self.use_double else filename
        filename = 'prioritized_' + filename if self.use_priority else filename
        
        return filename
        
    def save_weights(self, filename='local_weights.pth', path='weights'):
        filename = self.make_filename(filename)
        torch.save(self.qn_local.state_dict(), '{}/{}'.format(path, filename))
    
    def load_weights(self, filename='local_weights.pth', path='weights'):
        self.qn_local.load_state_dict(torch.load('{}/{}'.format(path, filename)))
    
    def summary(self):
        print('DQNAgent:')
        print('========')
        print('')
        print('Using Double:', self.use_double)
        print('Using Dueling:', self.use_dueling)
        print('Using Priority:', self.use_priority)
        print('Using Noise:', self.use_noise)
        print('')
        print(self.qn_local)