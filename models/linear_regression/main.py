
import torch
import torch.random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse

def generate_dataset(shape : tuple, low: int , high: int, seed: int = None) -> any  :
    
    N, dim = shape[0], shape[1]
    if seed is not None:
        torch.random.manual_seed(seed)    
        np.random.seed(seed=seed)
    sample = np.random.uniform(low, high, size=shape)
    X ,y = sample[:,:-1], sample[:,-1]
    
    X = torch.from_numpy(X.reshape(-1,1)).float()
    y = torch.from_numpy(y.reshape(-1,1)).float()

    return X, y
    

def create_model(num_parameters : int) -> torch.nn.Module:
    
    model = nn.Linear(num_parameters, 1)
    return model

def train_one_epoch(model: nn.Module, X_train, y_train, optimizer: optim.Optimizer, loss_fn):
    
    

    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    
    loss.backward()
    
    optimizer.step()
    
    optimizer.zero_grad()
    
    return loss
    
def train_model(model : nn.Module, X_train, y_train, hparams : dict, device : str = 'cpu', verbose = 'false') -> any:
   
    train_losses = []

    lr = hparams.get("learning_rate", 0.01)
    momentum = hparams.get("momentum", 0.2)
    epochs = hparams.get("epochs", 100)
    
    optimizer = optim.SGD(model.parameters(), lr, momentum)
    loss_fn = nn.MSELoss()
   
    for epoch in range(epochs):
        
        current_loss = train_one_epoch(model, X_train, y_train, optimizer, loss_fn)

        train_losses.append(round(current_loss.item(),4))
        if (verbose.lower() == 'true') and epoch % 10 == 0 :
            print(f'Epoch : {epoch} Current loss {current_loss}')

    return model, train_losses
   
def test_model(model: nn.Module, X_test, y_test, loss_fn):
    
    with torch.no_grad():
        
        predictions = model(X_test)
        loss = loss_fn(predictions, y_test)
        
    return loss, predictions
   
def print_configuration(hyperparameters : dict):
    print('Your configuration')
    for k,v in hyperparameters.items():
        print(f'{k} : {v}')

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
    
    args = parser.parse_args()
    
    hyperparameters = {'learning_rate': args.learning_rate,
                       'epochs': args.epochs,
                       'momentum': args.momentum}
    
    
    
    # Generate training dataset
    X_train, y_train = generate_dataset((100,2), 10, 15)
    
    model = create_model(1)
    
    
    #Train model
    model, train_losses = train_model(model, X_train,
                                      y_train,hyperparameters,
                                      verbose= args.verbose)
    #_ , training_predictions = test_model(model, X_train, y_train, nn.MSELoss())
    #Generate test dataset
    X_test, y_test = generate_dataset((30,2), 10, 15)
    #Test model against test dataset
    test_loss, predictions = test_model(model, X_test, y_test, nn.MSELoss())
    
    plt.figure(1)
    plt.plot(X_train, y_train, 'ro')
    plt.title('Training set')
    plt.plot(X_train, model(X_train).detach().numpy(), 'b-')
    plt.savefig('train_plot.jpg')
    
    plt.figure(2)
    plt.plot(X_test, y_test, 'ro')
    plt.title('Test set')
    plt.plot(X_test, model(X_test).detach().numpy(), 'b-')
    plt.savefig('test_plot.jpg')
    
    plt.figure(3)
    plt.plot(range(1,args.epochs+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Loss')
    plt.savefig('loss.jpg')
    
    
    
    


   
    
    