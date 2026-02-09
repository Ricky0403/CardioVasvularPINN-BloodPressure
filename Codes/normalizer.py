import torch

class MinMaxNormalizer:
    def __init__(self, data=None, method='global', device='cpu'):
        """
        method: 'global' (Isotropic) or 'column-wise' (Anisotropic/Individual)
        """
        self.min = None
        self.max = None
        self.device = device
        self.method = method
        
        if data is not None:
            self.fit(data)

    def fit(self, data):
        if self.method == 'global':
            self.min = data.min()
            self.max = data.max()
        
        elif self.method == 'column-wise':
            self.min = data.min(dim=0)[0]
            self.max = data.max(dim=0)[0]
            
        # Safety: Prevent division by zero
        diff = self.max - self.min
        # Create a mask for zero-range columns
        if torch.is_tensor(diff):
            diff[diff == 0] = 1.0 
        elif diff == 0:
            self.max += 1e-6

    def encode(self, data):
        if self.min is None: raise ValueError("Call .fit() first.")
        if data.device != self.device: self.to(data.device)
            
        # Formula: 2 * (x - min) / (max - min) - 1
        # Broadcasting handles the shapes automatically for both methods
        return 2 * (data - self.min) / (self.max - self.min) - 1

    def decode(self, data):
        if self.min is None: raise ValueError("Call .fit() first.")
        if data.device != self.device: self.to(data.device)
            
        return (data + 1) / 2 * (self.max - self.min) + self.min

    def to(self, device):
        self.device = device
        if self.min is not None:
            self.min = self.min.to(device)
            self.max = self.max.to(device)
        return self