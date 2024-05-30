import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.models.nn.common_layers import ResidualBlock

def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

class AdvancedCriticNetwork(nn.Module):
    """
    Advanced critic network designed to predict the value function for a given state input.
    Implements Batch Normalization, Leaky ReLU activations, Dropout, and Residual Connections for improved regularization.
    """
    def __init__(self, state_dim, hidden_sizes=None, dropout_rate=0.2):
        super(AdvancedCriticNetwork, self).__init__()
        hidden_sizes = hidden_sizes or [256, 256]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        input_dim = state_dim

        self.logger = configure_logger()  # Initialize the logger

        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_dim, size))
            self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            input_dim = size

        self.residual_block = ResidualBlock(input_dim, hidden_sizes[-1], dropout_rate)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        self.init_weights()

    def forward(self, x):
        if x.shape[1] != self.layers[0].in_features:
            self.logger.error(f"Incorrect input shape: got {x.shape[1]}, expected {self.layers[0].in_features}")
            raise ValueError(f"Incorrect input shape: got {x.shape[1]}, expected {self.layers[0].in_features}")

        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            if x.size(0) > 1:
                x = bn(x)
            x = F.leaky_relu(x)
            x = dropout(x)

        x = self.residual_block(x)
        value = self.value_head(x)
        return value

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0)
        nn.init.kaiming_normal_(self.value_head.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.value_head.bias, 0)

if __name__ == "__main__":
    state_dim = 10
    network = AdvancedCriticNetwork(state_dim)
    network.logger.info("Advanced Critic Network initialized successfully.")

    test_input = torch.rand(1, state_dim)
    try:
        value = network(test_input)
        network.logger.info(f"Computed value from the critic network: {value.item()}")
    except Exception as e:
        network.logger.error(f"Error testing network: {e}")