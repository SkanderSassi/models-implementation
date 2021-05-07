from model import ANN
import torch


if __name__ == '__main__':
    
    model = ANN([20,5,20,15,3])
        
    model.init_weights()
    
    
    test_tensor = torch.randn(32, 20)
    predictions = model.forward(test_tensor) 