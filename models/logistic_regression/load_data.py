import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def create_dataset(path: str, batch_size: int = 32, split_ratio: float = 0.2, seed= None):
    if seed is not None:
        torch.manual_seed(seed)
    transf = [transforms.ToTensor(), 
              transforms.Normalize((0.1307,), (0.3081,))
              ]
    train_dataset = datasets.MNIST(f'{path}', train=True,
                                      download=True, transform=transforms.Compose(transf),
                                      )
    test_dataset = datasets.MNIST(f'{path}', train=False,
                                    download=True, transform=transforms.Compose(transf),
                                    )

    train_dataset, val_dataset = data_utils.random_split(train_dataset, ( 50000, 10000 ))
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True) 
    valid_loader = DataLoader(val_dataset, batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True) 
    
    return train_loader, valid_loader, test_loader