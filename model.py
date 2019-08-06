import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, n_units=[64, 64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layers = {}
        
        for i, units in enumerate(n_units):
            if i == 0:
                # Input layer
                self.layers['fc0'] = nn.Linear(state_size, units)
            else:
                # Hidden layers
                self.layers['fc' + str(i)] = nn.Linear(n_units[i-1], units)
        
        # Output layer
        self.layers['fc' + str(i+1)] = nn.Linear(n_units[i], action_size)
            

    def forward(self, x):
        """Build a network that maps state -> action values."""
        
        n_items = len(self.layers)

        for i, (_, layer) in enumerate(self.layers.items()):
            if i < n_items - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
            
        return x
    
    def summary(self):
        for name, layer in self.layers.items():
            print('Layer {} => {}'.format(name, layer))