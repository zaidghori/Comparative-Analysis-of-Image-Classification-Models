import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Multilayer Perceptron model for image classification."""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.5, activation='relu'):
        """
        Initialize MLP model.
        
        Args:
            input_size: Size of flattened input (channels * height * width)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability
            activation: Activation function ('relu' or 'leaky_relu')
        """
        super(MLPModel, self).__init__()
        
        layers = []
        in_features = input_size
        
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        else:
            act_fn = nn.LeakyReLU
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                act_fn(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.net(x)