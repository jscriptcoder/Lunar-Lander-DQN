import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable


class NoisyLinear(nn.Linear):
    """Noisy linear layer with independent Gaussian noise:
        Extends Torch.nn.Linear according to the paper https://arxiv.org/abs/1706.10295,
        adding noise to the weights to aid efficient exploration. 
        The parameters of the noise are learned with gradient descent along with the remaining network weights.
        
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        sigma_init (float)
            Default: 0.017, 
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 sigma_init=0.017, 
                 bias=True):
        
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


class QNetwork(nn.Module):
    """Deep Q-Network model:
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        
    Args:
        state_size (int)
        action_size (int)
        seed (int)
        noisy (bool): Whether or not to add noisy layers
    
    Attributes:
        fc1 (Linear): Input layer (state_size, 32)
        fc2 (Linear | NoisyLinear): Hidden layer (32, 64)
        fc3 (Linear | NoisyLinear): Hidden layer (64, 128)
        fc4 (Linear | NoisyLinear): Output layer (128, action_size)
    """
    def __init__(self, state_size, action_size, seed, noisy=False):
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


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network model:
        See https://arxiv.org/abs/1511.06581
        
    Args:
        state_size (int)
        action_size (int)
        seed (int)
        noisy (bool): Whether or not to add noisy layers
    
    Attributes:
        features (PyTorch model):
            Input layer:  (state_size, 32)
            Hidden layer: (32, 64)
            Hidden layer: (64, 128)
        
        advantage (PyTorch model):
            Hidden layer: (128, 256)
            Output layer: (256, action_size)
            
        value (PyTorch model):
            Hidden layer: (128, 256)
            Output layer: (256, 1)
    """
    def __init__(self, state_size, action_size, seed, noisy=False):
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