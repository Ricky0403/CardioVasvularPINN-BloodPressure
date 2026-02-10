import  torch
import torch.nn as nn

class PINNModel(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(PINNModel, self).__init__()
        #Adding activation function
        self.activation = activation

        self.input_layer = nn.Linear(layers[0], layers[1])

        #Adding hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        #Output layer
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        #Physical parameter: viscosity 
        self.viscosity = nn.Parameter(torch.tensor([-5.65], dtype=torch.float32))
    
    def forward(self, x):
        out = self.activation(self.input_layer(x))
        
        # Residual Blocks with Skip Connections
        for layer in self.hidden_layers:
            residual = out
            out = self.activation(layer(out))
            
            # Only add skip connection if dimensions match
            if out.shape == residual.shape:
                out = out + residual 
        
        output = self.output_layer(out)
        return output