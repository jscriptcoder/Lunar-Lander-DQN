import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable


# Noisy linear layer with independent Gaussian noise
# See https://arxiv.org/abs/1706.10295
class NoisyLinear(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 sigma_init=0.017, bias=True):
        
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        
        
        if bias:
            self.sigma_bias = Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
            
        self.reset_parameters()

    def forward(self, input):
        weights = self.weight
        bias = self.bias
        
        if self.training:
            weights = self.weight + self.sigma_weight * Variable(self.epsilon_weight)
            
            if self.hasBias():
                bias = self.bias + self.sigma_bias * Variable(self.epsilon_bias)

        return F.linear(input, weights, bias)
    
    def hasBias(self):
        return self.bias is not None
    
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        init.uniform_(self.weight, -std, std)
        
        if self.hasBias():
            init.uniform_(self.bias, -std, std)


# Deep Q-Network model
# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, noisy=False):
        '''
        Architecture:
            Input layer:  (state_size, 32)
            Hidden layer: (32, 64)
            Hidden layer: (64, 128)
            Output layer: (128, action_size)
            
        Params
        ======
            state_size (int)
            action_size (int)
            seed (int)
            noisy (bool): whether or not to add noisy layers
        '''
        
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 32)
        
        if noisy:
            self.fc2 = NoisyLinear(32, 64)
        else:
            self.fc2 = nn.Linear(32, 64)
        
        if noisy:
            self.fc3 = NoisyLinear(64, 128)
        else:
            self.fc3 = nn.Linear(64, 128)
        
        if noisy:
            self.fc4 = NoisyLinear(128, action_size)
        else:
            self.fc4 = nn.Linear(128, action_size)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.fc4(x)


# Dueling Deep Q-Network model
# See https://arxiv.org/abs/1511.06581
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, noisy=False):
        '''
        Architecture:
            Input layer:  (state_size, 32)
            Hidden layer: (32, 64)
            Hidden layer: (64, 128)
            
            Advantage branch:
                Hidden layer: (128, 256)
                Output layer: (256, action_size)
            
            Value branch:
                Hidden layer: (128, 256)
                Output layer: (256, 1)
                
            Output: action_size
            
        Params
        ======
            state_size (int)
            action_size (int)
            seed (int)
            noisy (bool): whether or not to add noisy layers
        '''
        
        super(DuelingQNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # TODO: add noisy layers
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128) 
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
            
    def forward(self, state):
        x = self.features(state)
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean()