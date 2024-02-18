import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layer(input_dim, output_dim, batch_norm=True, activation_fn=nn.ReLU, dropout_prob=0.0):
    layers = [nn.Linear(input_dim, output_dim)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))
    layers.append(activation_fn())
    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))
    return nn.Sequential(*layers)

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, continuous=False, layer_sizes=[256, 128, 64], activation_fn=nn.ReLU, dropout_prob=0.0):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous

        self.layers = nn.ModuleList()
        last_size = input_size
        for size in layer_sizes:
            self.layers.append(create_layer(last_size, size, activation_fn=activation_fn, dropout_prob=dropout_prob))
            last_size = size

        self.fc_out = nn.Linear(last_size, output_size)
        if continuous:
            self.fc_std = nn.Linear(last_size, output_size)
            self.fc_std.weight.data.fill_(0.01)  # Small initialization for exploration

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        action_means = self.fc_out(x)
        if self.continuous:
            action_stds = F.softplus(self.fc_std(x)) + 1e-5  # Ensure std is positive
            return action_means, action_stds
        else:
            return F.softmax(action_means, dim=-1)
