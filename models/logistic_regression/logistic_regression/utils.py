import matplotlib.pyplot as plt
from numpy import arange

def plot_loss_acc(losses, accuracies, epochs):
    
    train_loss, val_loss = losses
    train_acc, val_acc = accuracies
    epochs_range = range(1, epochs +1)
    plt.figure(1)
    plt.plot(epochs_range, train_loss, 'r-', label='train')
    plt.plot(epochs_range, val_loss, 'b-', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Loss')
    plt.savefig('loss.jpg')
    
    plt.figure(2)
    plt.plot(epochs_range, train_acc, 'r-', label='train')
    plt.plot(epochs_range, val_acc, 'b-', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training/Validation Accuracy')
    plt.savefig('accuracy.jpg')