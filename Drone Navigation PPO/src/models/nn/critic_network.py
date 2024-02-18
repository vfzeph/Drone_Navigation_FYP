import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1, dropout_rate=0.2, use_batch_norm=True, activation_fn=F.relu):
        super(ValueNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn
        
        # Optional: Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes]) if use_batch_norm else None
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation_fn(x)
            x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x
