import argparse
import torch
from load_data import create_dataset


def check_classes(loader):
    class_map = {class_label : 0 for class_label in range(10)}
    with torch.no_grad():
        for i, data in enumerate(loader):
            _, labels = data
            for label in labels:
                class_map[label.item()] += 1
    return class_map

def create_lr_model(in, out):
    return torch.nn.Linear(in, out)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a linear and test linear a linear regression model')
    parser.add_argument('-lr','--learning_rate', help='Set the learning rate (default 0.01)',
                        type=float, default=1e-2, required=False )
    parser.add_argument('-e', '--epochs',help="Number of training epochs (default 100)",
                        type=int, default=100, required=False)
    parser.add_argument('-mm','--momentum',help="Momentum (default 0)", type=float,
                        default = 0, required = False)
    parser.add_argument('-v','--verbose', help="Display losses for epochs (default False)"
                        , type=str, default= False, required = False)
    
    train_loader, val_loader, test_loader = create_dataset('../datasets')
    