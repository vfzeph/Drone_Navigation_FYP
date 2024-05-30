import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers and a skip connection.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if out.size(0) > 1:
            out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out += residual
        out = F.leaky_relu(out)
        return out

class AttentionLayer(nn.Module):
    """
    Simple attention layer to focus on important parts of the input state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, x):
        attention_weights = torch.matmul(x, self.context_vector)
        attention_weights = F.softmax(attention_weights, dim=0)
        attended_state = x * attention_weights.unsqueeze(1)
        return attended_state

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(128 + action_dim, 128),  # Adjusted input dimension
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state, next_state, action):
        state_feat = self.encoder(state)
        next_state_feat = self.encoder(next_state)
        
        # Ensure action is 2D
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Ensure the features are 2D
        if state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)
        if next_state_feat.dim() == 1:
            next_state_feat = next_state_feat.unsqueeze(0)

        # Ensure action is the correct size (batch_size, action_dim)
        if action.size(0) != state_feat.size(0):
            action = action.expand(state_feat.size(0), -1)

        # Adjust dimensions for concatenation
        state_action_feat = torch.cat((state_feat, action), dim=1)
        if state_action_feat.size(1) != 128 + action.size(1):
            padding = torch.zeros(state_action_feat.size(0), (128 + action.size(1)) - state_action_feat.size(1))
            state_action_feat = torch.cat((state_action_feat, padding), dim=1)

        # Print shapes for debugging
        print(f"state_feat shape: {state_feat.shape}")
        print(f"next_state_feat shape: {next_state_feat.shape}")
        print(f"action shape: {action.shape}")
        print(f"state_action_feat shape: {state_action_feat.shape}")

        action_pred = self.inverse_model(torch.cat((state_feat, next_state_feat), dim=1))
        next_state_pred = self.forward_model(state_action_feat)

        # Print shapes for debugging
        print(f"action_pred shape: {action_pred.shape}")
        print(f"next_state_pred shape: {next_state_pred.shape}")

        return state_feat, next_state_feat, action_pred, next_state_pred

    def intrinsic_reward(self, state, next_state, action):
        state_feat, next_state_feat, _, next_state_pred = self.forward(state, next_state, action)
        reward = torch.mean((next_state_feat - next_state_pred) ** 2, dim=1).item()
        return reward
