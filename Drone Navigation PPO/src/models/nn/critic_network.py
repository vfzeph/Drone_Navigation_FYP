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
        
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size, track_running_stats=True) for size in hidden_sizes])
        else:
            self.batch_norms = None
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm and x.size(0) > 1:  # Check if batch size is greater than 1
                x = self.batch_norms[i](x)
            x = self.activation_fn(x)
            x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x
    
    def test_value_network():
        input_size = 10  # Example input size
        output_size = 1  # Example output size
        test_input = torch.randn(5, input_size)  # Batch size of 5
        network = ValueNetwork(input_size, hidden_sizes=[128, 64], output_size=output_size)
        output = network(test_input)
        assert output.shape == (5, output_size), "Unexpected output shape"
