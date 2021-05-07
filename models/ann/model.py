import math
import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, dimensions: list, ) -> None:
        super(ANN, self).__init__()
        self.dims = dimensions
        self.depth = len(dimensions)-1
        self.weights = nn.ParameterDict()
        for i in range(self.depth):
            self.weights[f'W{i}'] = nn.parameter.Parameter(torch.Tensor(self.dims[i], self.dims[i+1]))
            self.weights[f'b{i}'] = nn.parameter.Parameter(torch.Tensor(self.dims[i+1]))
        
        
    def init_weights(self):
        
        for param in self.parameters():
            dim = param.size(-1)
            stdv = 1.0 / math.sqrt(dim)
            param.data.uniform_(-stdv, stdv)
        
    
    def show_weights(self):
        
        for param in self.parameters():
            print(param)
        
    def forward(self, x):
        
        a_prev = x
        for i in range(self.depth-1):
            a_prev = a_prev @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            a_prev = torch.relu(a_prev)
        predictions = a_prev @ self.weights[f'W{i+1}'] + self.weights[f'b{i+1}']
        return predictions
        