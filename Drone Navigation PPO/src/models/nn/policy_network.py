import torch
import torch.nn as nn

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, layer_sizes=[256, 128, 64], activation_fn=nn.ReLU, dropout_prob=0.0):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_dim, layer_sizes[0]))
        
        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(activation_fn())
            self.layers.append(nn.Dropout(dropout_prob))
            
            # Add batch normalization except for the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))

        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-1], action_dim)
        if not continuous:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.continuous:
            # For continuous action space, return mean and standard deviation
            action_mean = torch.tanh(self.output_layer(x[:, :self.action_dim]))  # Assuming the first part is mean
            action_std = torch.exp(self.output_layer(x[:, self.action_dim:]))  # Assuming the second part is log std
            return action_mean, action_std
        else:
            # For discrete action space, return action probabilities
            action_probs = self.softmax(self.output_layer(x))
            return action_probs

    def disable_batch_norm_inference(self):
        # Set batch normalization layers to evaluation mode during inference
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
