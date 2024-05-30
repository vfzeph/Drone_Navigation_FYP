import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.models.nn.common_layers import ResidualBlock, AttentionLayer

def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = configure_logger()

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False, hidden_sizes: list = [128, 128], dropout_rate: float = 0.1):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        input_dim = state_dim

        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_dim, size))
            self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            input_dim = size

        self.residual_block = ResidualBlock(input_dim, hidden_sizes[-1], dropout_rate)
        self.attention = AttentionLayer(hidden_sizes[-1], hidden_sizes[-1])

        if self.continuous:
            self.mean = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

        self.init_weights()

    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            if x.size(0) > 1:
                x = bn(x)
            x = F.leaky_relu(x)
            x = dropout(x)

        x = self.residual_block(x)
        x = self.attention(x)

        if self.continuous:
            action_mean = self.mean(x)
            action_std = self.log_std.exp()
            return action_mean, action_std
        else:
            action_probs = F.softmax(self.action_head(x), dim=-1)
            return action_probs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    logger.info("Initializing and testing the AdvancedPolicyNetwork.")
    state_dim = 10
    action_dim = 4
    network = AdvancedPolicyNetwork(state_dim, action_dim, continuous=True)
    test_input = torch.rand(1, state_dim)
    network.eval()
    with torch.no_grad():
        action_output = network(test_input)
        if isinstance(action_output, tuple):
            logger.info(f"Action outputs: mean={action_output[0]}, std={action_output[1]}")
        else:
            logger.info(f"Action probabilities: {action_output}")
