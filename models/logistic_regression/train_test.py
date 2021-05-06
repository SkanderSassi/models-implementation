import torch.nn as nn

def calculate_loss(data, loss_fn, device = None):
    inputs, labels = data 
    
       
def train_model(model, optimizer, hparams, train_loader, loss_fn ,device = None ):
    
    epochs = hparams.get("epochs", 10)
    batch_size = hparams.get("batch_size")
    
    model.train()
    
    for epoch in range(1, epochs+1):
        
        train_loss = 0.0
        for batch_id, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            
            optimizer.zero_grad()

            
            predictions = model(inputs)
            loss = loss_fn(labels)
            
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                print(f'Epoch {epoch} Batch {batch_id} / {batch_size}  ') 
    
    

    
        
    
    pass
    