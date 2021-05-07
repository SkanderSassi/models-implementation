import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
INPUT_DIM = 28*28

def calculate_val_loss(model, loader, loss_fn, device = None):
    
    with torch.no_grad():
        loss = 0.0
        for batch_id, data in enumerate(loader):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.reshape(-1, INPUT_DIM)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            
        return loss.item()
def calculate_train_loss(model ,data, loss_fn, device = None):
    
    inputs, labels = data[0].to(device), data[1].to(device) 
    
    inputs = inputs.reshape(-1, INPUT_DIM)
    
    predictions = model(inputs)
    loss = loss_fn(predictions, labels)
    
    return loss
    
def calculate_acc(model : torch.nn.Module, test_loader : DataLoader, metric : str = 'mean_accuracy', device = None):
    # Need to implement other metrics
    correct = 0
    accuracy = 0.0
    data_len = len(test_loader.dataset)
    with torch.no_grad():
        model.eval()
        for _, data in enumerate(test_loader):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.reshape(-1, INPUT_DIM)
            predictions = model(inputs)
            
            
            correct += torch.sum(predictions.argmax(axis=1) == labels)
        if metric == 'mean_accuracy':
            accuracy = round(float(correct / data_len), 2)
            
    
    return accuracy

    
    
    
       
def train_model(model : torch.nn.Module, optimizer : torch.optim.Optimizer, hparams : dict, train_loader : DataLoader, 
                valid_loader : DataLoader, loss_fn ,device = None ):
    
    epochs = hparams.get("epochs", 10)
    batch_size = hparams.get("batch_size")
    device = 'cuda:0' if (torch.cuda.is_available() and device == 'gpu') else 'cpu'
    
    print(f'Using {device}')
    total_steps = len(train_loader)

    train_losses = list()
    train_accuracies = list()
    val_losses = list()
    val_accuracies = list()
    
    model.to(device)
    for epoch in range(1, epochs+1):
        
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        running_loss = 0.0
        for batch_id, data in enumerate(train_loader, 0):
            
            optimizer.zero_grad()
            loss = calculate_train_loss(model ,data, loss_fn, device)
            loss.backward()
            optimizer.step()
            
            if batch_id % 100 == 0:
                print(f'Epoch : [{epoch} / {epochs}] Batch : [{batch_id} / {total_steps}] Loss: {loss.item()}')
        
        val_acc = calculate_acc(model, valid_loader, metric='mean_accuracy', device=device )    
        
        val_loss = round(calculate_val_loss(model, valid_loader, loss_fn, device=device), 3)
        train_acc = calculate_acc(model, train_loader, metric = 'mean_accuracy', device=device)
        print(f'Epoch: {epoch} Train acc {train_acc} Validation loss: {val_loss} Validation acc: {val_acc}') 
        

        print('-'*60)
        
        train_losses.append(loss.item())
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    model.to('cpu')
    return model, train_losses, val_losses, train_accuracies, val_accuracies



    