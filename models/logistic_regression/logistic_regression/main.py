import argparse
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from load_data import create_dataset
from train_test import train_model, calculate_acc
from utils import plot_loss_acc

INPUT_DIM = 28 * 28
NUM_CLASSES = 10



def check_classes(loader):
    class_map = {class_label : 0 for class_label in range(10)}
    with torch.no_grad():
        for i, data in enumerate(loader):
            _, labels = data
            for label in labels:
                class_map[label.item()] += 1
    return class_map

def create_lr_model(in_dim, out_dim):
    model = torch.nn.Linear(in_dim, out_dim)

    return model


def print_configuration(hyperparameters : dict):
    print('Your configuration')
    for k,v in hyperparameters.items():
        print(f'{k} : {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a linear and test linear a linear regression model')
    parser.add_argument('-lr','--learning_rate', help='Set the learning rate (default 0.01)',
                        type=float, default=1e-4, required=False )
    parser.add_argument('-e', '--epochs',help="Number of training epochs (default 100)",
                        type=int, default=100, required=False)
    parser.add_argument('-mm','--momentum',help="Momentum (default 0)", type=float,
                        default = 0, required = False)
    parser.add_argument('-v','--verbose', help="Display losses for epochs (default False)"
                        , type=str, default= False, required = False)
    parser.add_argument('-bs','--batch_size', help="Set batch size"
                        , type=int, default= 32, required = False)
    args = parser.parse_args()

    
    hyperparameters = {'learning_rate': args.learning_rate,
                       'epochs': args.epochs,
                       'momentum': args.momentum,
                       'batch_size': args.batch_size}
    
    print_configuration(hyperparameters)
    
    train_loader, val_loader, test_loader = create_dataset('../datasets', hyperparameters.get('batch_size', 32))
    
    model = create_lr_model(INPUT_DIM, NUM_CLASSES)
    optimizer = SGD(model.parameters(), hyperparameters.get('learning_rate', 1e-4), 
                    momentum=hyperparameters.get("momentum", 0))
    loss_fn = CrossEntropyLoss()
    
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, optimizer, hyperparameters, 
                                                                                    train_loader, val_loader, loss_fn,
                                                                                    'gpu')
    
    print('Accuracy on test set', calculate_acc(model, test_loader=test_loader, metric='mean_accuracy', device='cpu'))
    
    plot_loss_acc((train_losses, val_losses), (train_accuracies, val_accuracies), hyperparameters.get('epochs'))
    