import torch
import torch.nn as nn
import numpy as np

class FourierFeatureTransform(nn.Module):
    def __init__(self, in_features, mapping_size, scale=1.0):
        super(FourierFeatureTransform, self).__init__()
        
        self.in_features = in_features
        self.mapping_size = mapping_size
        
        # Initialize the random weight matrix B
        B = torch.randn((in_features, mapping_size)) * scale
        
        self.register_buffer('B', B)

    def forward(self, x):
        # x shape: (Batch, in_features)
        # B shape: (in_features, mapping_size)
        # Projection shape: (Batch, mapping_size)
        x_proj = 2.0 * np.pi * x @ self.B
        
        # Output shape: (Batch, 2 * mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class PINNModel(nn.Module):
    def __init__(self, layers, fourier_mapping_size=32, fourier_scale=1.0, activation=nn.Tanh()):
        super(PINNModel, self).__init__()
        self.activation = activation

        #Fourier Transform Layer
        self.fourier = FourierFeatureTransform(in_features=layers[0], 
                                               mapping_size=fourier_mapping_size, 
                                               scale=fourier_scale)
        
        # The output of Fourier layer
        fourier_out_dim = fourier_mapping_size * 2

        #Input Layer
        self.input_layer = nn.Linear(fourier_out_dim, layers[1])

        #Hidden Layers 
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        #Output Layer
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        
        #Viscosity
        self.viscosity = nn.Parameter(torch.tensor([-5.65], dtype=torch.float32))
    
    def forward(self, x):
        # Pass raw coordinates through Fourier mapping
        x_fourier = self.fourier(x)
        
        # Pass the Fourier features into the standard network
        out = self.activation(self.input_layer(x_fourier))
        
        # Residual Blocks with Skip Connections
        for layer in self.hidden_layers:
            residual = out
            out = self.activation(layer(out))
            if out.shape == residual.shape:
                out = out + residual 
        
        output = self.output_layer(out)
        return output