import torch
import pandas as pd
from vgg_16 import Network
from torchvision.transforms import ToTensor
from torchvision import datasets

def initialize_network(use_svhn, OCL_sw, kasami_path='kasami_10X255.csv', device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Network(use_svhn=use_svhn).to(device)
    if OCL_sw:
        kasami = pd.read_csv(kasami_path)
        kasami_tensor = torch.FloatTensor(kasami.values)
        kasami_tensor = kasami_tensor.to(device)
        network.fc_k.bias.data.zero_()
        network.fc_k.weight.data = kasami_tensor
    return network

def initialize_kasami_tensor(kasami_file='kasami_10X255.csv'):
    kasami = pd.read_csv(kasami_file)
    kasami_tensor = torch.FloatTensor(kasami.values)
    return kasami_tensor


def initialize_dataset(split, use_svhn):
    if use_svhn:
        DATA = datasets.SVHN
        initialize_dataset = DATA(root='./data', split=split, download=True, transform=ToTensor())
    else:
        DATA = datasets.MNIST
        initialize_dataset = DATA(root='./data', train=True, download=True, transform=ToTensor())
    return initialize_dataset


def get_model_save_path(use_svhn, OCL_sw, model_dir='models/'):
    dataset_name = 'SVHN' if use_svhn else 'MNIST'
    model_tag = 'OCL' if OCL_sw else 'NoOCL'
    model_save_path = model_dir + dataset_name + '_' + model_tag + '.pt'
    return model_save_path


def load_checkpoint_to_model(network, PATH, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    network.load_state_dict(checkpoint)
    return network