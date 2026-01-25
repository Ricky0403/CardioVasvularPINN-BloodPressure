import  torch
import torch.nn as nn

class PINNModel(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(PINNModel, self).__init__()
        #Create a list to hold the layers
        self.layers = nn.ModuleList()
        #Adding activation function
        self.activation = activation

        #Adding hidden layers
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        #Output layer
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        #Physical parameter: viscosity (as a learnable parameter)
        self.viscosity = nn.Parameter(torch.tensor([0.0035], dtype=torch.float32))
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        output = self.output_layer(x)
        return output