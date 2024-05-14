import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, layer_sizes=[256, 128, 64], activation_fn=nn.ReLU, dropout_prob=0.1):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_dim, layer_sizes[0]))
        
        # Dynamic dropout implementation
        dropout_layers = [nn.Dropout(p=(i + 1) * dropout_prob / len(layer_sizes)) for i in range(len(layer_sizes))]

        # Hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(activation_fn())
            self.layers.append(dropout_layers[i - 1])
            self.layers.append(nn.BatchNorm1d(layer_sizes[i]))
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        # Output layer configuration for continuous/discrete actions
        if continuous:
            self.mean_layer = nn.Linear(layer_sizes[-1], action_dim)
            self.std_layer = nn.Linear(layer_sizes[-1], action_dim)
        else:
            self.output_layer = nn.Linear(layer_sizes[-1], action_dim)
            self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x)

        if self.continuous:
            mean = self.mean_layer(x)
            std = torch.exp(self.std_layer(x))  # Ensure non-negative std deviation
            return mean, std
        else:
            return self.softmax(self.output_layer(x))

