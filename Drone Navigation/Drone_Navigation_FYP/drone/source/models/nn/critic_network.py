import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], output_size=1, dropout_rate=0.1, use_batch_norm=True, activation_fn=F.relu):
        super(CriticNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn

        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes])
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.output_layer = nn.Linear(prev_size, output_size)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm and x.size(0) > 1:
                x = self.batch_norms[i](x)
            x = self.activation_fn(x)
            x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x
