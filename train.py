
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
import copy
from vgg_16 import Network
from tqdm.auto import tqdm
import sys
import os

def initialize_datasets(use_svhn):
    if use_svhn:
        from torchvision.datasets import SVHN as DATA
    else:
        from torchvision.datasets import MNIST as DATA

    if use_svhn:
        train_set = DATA(root='./data', split='train', download=True, transform=ToTensor())
    else:
        train_set = DATA(root='./data', train=True, download=True, transform=ToTensor())

    return train_set

def split_train_valid(train_set, valid_size, valid_shuffle):
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if valid_shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler

def initialize_network(use_svhn, device, OCL_sw):
    network = Network(use_svhn=use_svhn).to(device)

    if OCL_sw:
        kasami = pd.read_csv('kasami_10X255.csv')
        kasami_tensor = torch.FloatTensor(kasami.values)
        kasami_tensor = kasami_tensor.to(device)
        network.fc_k.bias.data.zero_()
        network.fc_k.weight.data = kasami_tensor

    return network


def initialize_kasami_tensor(kasami_file='kasami_10X255.csv'):
    kasami = pd.read_csv(kasami_file)
    kasami_tensor = torch.FloatTensor(kasami.values)
    return kasami_tensor

def train_network(network, loader_train, loader_val, epochs, loss_function_cross, loss_function_mse, optimizer, device, alpha, OCL_sw, model_save_path, save_interval):
    batch_size = loader_train.batch_size
    val_batch_size = loader_val.batch_size
    kasami_tensor = initialize_kasami_tensor()
    kasami_tensor = kasami_tensor.to(device)
    steps_counter = 0
    if OCL_sw:
        network.fc_k.bias.data.zero_()
        network.fc_k.weight.data = kasami_tensor
        network.fc_k.weight.requires_grad = False
        network.fc_k.bias.requires_grad = False
    network.train()

    # Initialize progress bars for epochs
    epochs_bar = tqdm(range(epochs), desc=f'Epochs Progress')

    for epoch in epochs_bar:
        num_correct_now = 0
        loss = 0.0
        
        # Initialize progress bars for batches
        batches_bar = tqdm(loader_train, desc=f'Batches Progress Loss: {loss:.4f} Correct: {num_correct_now}/{batch_size}',position=0, leave=True)

        for batch in batches_bar:
            steps_counter += 1

            # Get batch data
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            [kasami_preds, outputs] = network(images)

            # Cross entropy loss
            loss_cross = loss_function_cross(outputs, labels)

            # MSE loss
            kasami_target = kasami_tensor[labels]
            loss_mse = loss_function_mse(kasami_preds, kasami_target)
            
            # Total loss
            loss = alpha * loss_mse + (1 - alpha) * loss_cross

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss and number of correct predictions
            loss = loss.item()
            preds = torch.argmax(outputs, dim=1)
            num_correct_now = preds.eq(labels).sum().item()

            # Update progress bars
            batches_bar.set_description(f'Batches Progress Loss: {loss:.4f} Correct: {num_correct_now}/{batch_size}')
            batches_bar.refresh()

            # Save model every save_interval steps
            if steps_counter % save_interval == 0:
                steps_save_path = model_save_path[:-3] + f'_{steps_counter}.pt'
                torch.save(network.state_dict(), steps_save_path)





        # Validate network every epoch
        # Set network to evaluation mode
        with torch.no_grad():
            running_loss = 0.0
            num_correct = 0
            loss = 0.0
            num_correct_now = 0
            num_batches = len(loader_val)
            num_images = num_batches * val_batch_size

            # Initialize progress bars for batches
            eval_batches_bar = tqdm(loader_val, desc=f'Val Batches Progress Loss: {loss:.4f} Correct: {num_correct_now}/{batch_size}  Total Correct: {num_correct}/{num_images}', position=0, leave=True)
            for batch in eval_batches_bar:
                # Get batch data
                images = batch[0].to(device)
                labels = batch[1].to(device)

                # Forward pass
                [kasami_preds, outputs] = network(images)

                # Cross entropy loss
                loss_cross = loss_function_cross(outputs, labels)

                # MSE loss
                kasami_target = kasami_tensor[labels]
                loss_mse = loss_function_mse(kasami_preds, kasami_target)

                # Total loss
                loss = alpha * loss_mse + (1 - alpha) * loss_cross

                # Update running loss and number of correct predictions
                preds = torch.argmax(outputs, dim=1)
                num_correct_now = preds.eq(labels).sum().item()
                num_correct += num_correct_now
                running_loss += loss.item()

                # Update progress bars
                eval_batches_bar.set_description(f'Val Batches Progress Loss: {loss:.4f} Correct: {num_correct_now}/{batch_size}  Total Correct: {num_correct}/{num_images}') 
                eval_batches_bar.refresh()


        if epoch == 0:
            best_acc = num_correct
            best_epoch = epoch
            best_model = copy.deepcopy(network.state_dict())
            torch.save(best_model, model_save_path)
            print(f'Best model saved at epoch {best_epoch} with accuracy {best_acc/num_images} and loss {running_loss/num_batches}')
        else:
            if num_correct > best_acc:
                best_acc = num_correct
                best_epoch = epoch
                best_model = copy.deepcopy(network.state_dict())
                torch.save(best_model, model_save_path)
                print(f'Best model saved at epoch {best_epoch} with accuracy {best_acc/num_images} and loss {running_loss/num_batches}')





def main(
        use_svhn=False,
        valid_size=0.2,
        valid_shuffle=True,
        batch_size=64,
        epochs=10,
        learning_rate=0.01,
        OCL_sw=True,
        alpha=0.5,
        seed=None,
        device='cpu',
        model_save_path='./models/best_model.pt',
        save_interval=100
        ):
    
    #Set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    #Initialize datasets
    train_set = initialize_datasets(use_svhn)

    #Split train and validation sets
    train_sampler, val_sampler = split_train_valid(train_set, valid_size, valid_shuffle)

    #Initialize dataloaders
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    loader_val = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

    #Initialize network
    network = initialize_network(use_svhn, device, OCL_sw)

    #Initialize loss functions
    loss_function_cross = nn.CrossEntropyLoss()
    loss_function_mse = nn.MSELoss()

    #Initialize optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    #Train network
    train_network(
        network,
        loader_train,
        loader_val,
        epochs,
        loss_function_cross,
        loss_function_mse,
        optimizer,
        device,
        alpha,
        OCL_sw,
        model_save_path,
        save_interval
        )


               

if __name__ == '__main__':
    use_svhn = False
    valid_size = 0.5
    valid_shuffle = True
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    OCL_sw = True
    alpha = 0.5
    seed = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_interval = 100

    if use_svhn:
        dataset_name = 'SVHN'
    else:
        dataset_name = 'MNIST'
    
    if OCL_sw:
        model_save_path = f'./models/{dataset_name}_OCL.pt'
    else:
        model_save_path = f'./models/{dataset_name}_noOCL.pt'
        
    
    main(
        use_svhn=use_svhn,
        valid_size=valid_size,
        valid_shuffle=valid_shuffle,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        OCL_sw=OCL_sw,
        alpha=alpha,
        seed=seed,
        device=device,
        model_save_path=model_save_path,
        save_interval=save_interval
        )