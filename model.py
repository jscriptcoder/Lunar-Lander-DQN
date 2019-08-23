import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable


# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            mu_range = math.sqrt(3 / self.in_features)
            init.uniform_(self.weight, -mu_range, mu_range)
            init.uniform_(self.bias, -mu_range, mu_range)
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        if self.training:
            return F.linear(input, 
                            self.weight + self.sigma_weight * Variable(self.epsilon_weight), 
                            self.bias + self.sigma_bias * Variable(self.epsilon_bias))
        else:
            return F.linear(input, self.weight, self.bias)

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)

class QNetwork(nn.Module):
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
    def __init__(self, state_size, action_size, seed, noisy=False):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
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