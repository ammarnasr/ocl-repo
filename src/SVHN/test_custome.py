import torch
from torch.utils.data import Dataset, DataLoader
from vgg_16 import Network
from tqdm import tqdm
from torch.nn.functional import one_hot
# from svhn_custome_dataset import SVHN_Dataset
import sys
from torchvision.transforms import ToTensor
from torchvision.datasets import SVHN as DATA
import numpy as np
from torch.nn.functional import one_hot

ckpt_name = "./models/VGG16_0.002_SVHN_True_200_epochs.pt"
n = len(sys.argv)

if n == 2:
    name_input = sys.argv[1]
    ckpt_name = name_input


def get_num_correct(p, l):
    compared = p + l
    corrects = torch.sum(compared == 2, axis=1)
    corrects_sum = torch.sum(corrects).item()
    return corrects_sum


def mergimages(x, y, overlap):
    if overlap == 1:
        new_image = x + y
        return new_image
    batch_size = x.shape[0]
    c_normal = x.shape[1]
    h_noraml = x.shape[2]
    w_normal = x.shape[3]
    w_new = int((2 - overlap) * w_normal)
    padding = w_new - w_normal
    zero_padding_array = torch.zeros(batch_size, c_normal, h_noraml, padding)
    left_image = torch.cat((x, zero_padding_array), dim=3)
    right_image = torch.cat((zero_padding_array, y), dim=3)
    new_image = left_image + right_image
    return new_image


def get_preds_from_outputs(outputs, num_images):
    preds = []
    for i in range(len(num_images)):
        n = num_images[i].item()
        tk = torch.topk(outputs[i], n)[1]
        tk_oh = one_hot(tk, num_classes=10)
        tk_oh_sum = torch.sum(tk_oh, dim=0).tolist()
        preds.append(tk_oh_sum)

    preds = torch.tensor(preds, device=device)
    return preds


# x = SVHN_Dataset('org_data/test/digitStruct.mat','org_data/test')
x = DATA(root='./data', split='test', download=True, transform=ToTensor())

batch_size = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loader_test = DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=0)
loader_test = DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=0)
PATH = ckpt_name
checkpoint = torch.load(PATH, map_location=torch.device(device))
network = Network().to(device)
network.load_state_dict(checkpoint['model_state_dict'])
num_correct = 0
n_all_images = 0
accuracy = 0
network.eval()

# with torch.no_grad():
#   for batch in loader_test:
#     images = batch['image']
#     labels = batch['labels']
#     images = images.to(device).float()
#     labels = labels.to(device)
#     num_images =torch.sum(labels, dim=1)
#     n_all_images += torch.sum(num_images).item()
#     [kasami_preds, outputs] = network(images)

#     preds = get_preds_from_outputs(outputs, num_images)
#     num_correct_now = get_num_correct(preds,labels)
#     num_correct += num_correct_now
# accuracy = num_correct / (n_all_images)
# print("Test Accuarcy of Multi-Label", PATH, "is : ", accuracy * 100, "% no Overlap")

# with torch.no_grad():
#   for batch in loader_test:
#     batch_count += 1
#     # images = batch['image']
#     # labels = batch['labels']
#     images = batch[0]
#     labels = batch[1]
#     labels = one_hot(labels,  num_classes=10)
#     # print(type(batch),'---',labels[0])
#     images = images.to(device).float()
#     labels = labels.to(device)
#     num_images =torch.sum(labels, dim=1)
#     n_all_images += torch.sum(num_images).item()
#     [kasami_preds, outputs] = network(images)

#     preds = get_preds_from_outputs(outputs, num_images)
#     num_correct_now = get_num_correct(preds,labels)
#     num_correct += num_correct_now
# accuracy = num_correct / (n_all_images)
# print("Test Accuarcy of Multi-Label", PATH, "is : ", accuracy * 100, "% no Overlap")


with torch.no_grad():
    for batch in loader_test:
        current_batch_szie = batch[0].shape[0]
        rand_indx = np.arange(current_batch_szie)
        np.random.shuffle(rand_indx)
        # images = batch['image']
        # labels = batch['labels']
        images_1 = batch[0]
        images_2 = images_1[rand_indx]
        # images = torch.cat([images_1, images_2], dim=2)
        images = mergimages(images_1, images_2, .1)

        labels = batch[1]
        labels_rand = labels[rand_indx]
        labels = one_hot(labels, num_classes=10)
        labels_rand = one_hot(labels_rand, num_classes=10)
        labels = labels + labels_rand
        for i in range(current_batch_szie):
            labels[i] = labels[i] / max(labels[i])
            labels[i] = np.ceil(labels[i])
        # print(type(batch),'---',labels[0])
        images = images.to(device).float()
        labels = labels.to(device)
        num_images = torch.sum(labels, dim=1)
        n_all_images += torch.sum(num_images).item()
        [kasami_preds, outputs] = network(images)

        preds = get_preds_from_outputs(outputs, num_images)
        num_correct_now = get_num_correct(preds, labels)
        num_correct += num_correct_now
accuracy = num_correct / (n_all_images)
print("Test Accuarcy of Multi-Label", PATH, "is : ", accuracy * 100, "% no Overlap")

