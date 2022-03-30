import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import SVHN as DATA
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
import copy
from vgg_16 import Network
from tqdm import tqdm
import sys

epochs = 10
OCL_sw = True
n = len(sys.argv)

if n == 3 :  
  ocl_input = sys.argv[1]
  if ocl_input == '0' :
    OCL_sw = False
  epochs_input = sys.argv[2]
  epochs = int(epochs_input)


alpha = 0.0;
valid_size = 0.2
valid_shuffle = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The Device is ', device)
loss_function_cross = nn.CrossEntropyLoss()
loss_function_mse = nn.MSELoss()

train_set = DATA(root='./data', split='train', download=True, transform=ToTensor())
num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
if valid_shuffle == True:
    np.random.seed(1)
    np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
lr = 1e-4
batch_size = 16
Wdecay = 0 * 1e-4
networkName = "VGG16"
datasetName = "SVHN"

network = Network().to(device)

##########OCL##########################
kasami = pd.read_csv('kasami_10X255.csv')
kasami_tensor = torch.FloatTensor(kasami.values)
kasami_tensor = kasami_tensor.to(device)
###initializing the weights and bias for fc_k
if OCL_sw:
    network.fc_k.bias.data.zero_()
    network.fc_k.weight.data = kasami_tensor
#########################################

loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
loader_val = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)
optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=Wdecay)

print("Length of Train Loader: ", len(loader_train))
print("Length of Valid Loader: ", len(loader_val))

best_acc = 0
best_epoch = 0

for epoch in tqdm(range(epochs), desc = 'Epochs Progress'):
    num_correct = 0
    running_loss = 0.0
    i = 0
    network.train()

    ##########OCL##########################
    if OCL_sw:
        network.fc_k.weight.requires_grad = False
        network.fc_k.bias.requires_grad = False
    #######################################

    for batch in loader_train:
        i += 1
        images = batch[0].to(device)
        labels = batch[1].to(device)
        kasami_target = kasami_tensor[labels]

        [kasami_preds, outputs] = network(images)

        loss_cross = loss_function_cross(outputs, labels)
        loss_mse = loss_function_mse(kasami_preds, kasami_target)
        loss = alpha * loss_mse + (1 - alpha) * loss_cross

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, dim=1)
        num_correct_now = preds.eq(labels).sum().item()
        num_correct += num_correct_now
        running_loss += loss.item()

    network.eval()
    with torch.no_grad():
        i = 0
        running_loss = 0.0
        num_correct = 0
        for batch in loader_val:
            i = i + 1
            images = batch[0].to(device)
            labels = batch[1].to(device)
            kasami_target = kasami_tensor[labels]

            [kasami_preds, outputs] = network(images)

            loss_cross = loss_function_cross(outputs, labels)
            loss_mse = loss_function_mse(kasami_preds, kasami_target)
            loss = alpha * loss_mse + (1 - alpha) * loss_cross

            preds = torch.argmax(outputs, dim=1)
            num_correct_now = preds.eq(labels).sum().item()
            num_correct += num_correct_now
            running_loss += loss.item()

        len_valid = len(loader_val) * batch_size
        valid_accuracy = num_correct / len_valid
        if valid_accuracy > best_acc:
            best_epoch = epoch
            best_acc = valid_accuracy
            best_model = copy.deepcopy(network)
            print("New Best Accuracy of", best_acc * 100, "% At Epoch ", best_epoch, "")


print("Training Complete, Best Validation Accuracy: ", best_acc)
PATH = './models/' + networkName + "_" + str(lr) + "_" + datasetName + "_" + str(OCL_sw) + ".pt"
torch.save({'model_state_dict': best_model.state_dict()}, PATH)