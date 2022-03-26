import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import SVHN as DATA
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from vgg_16 import Network


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


ckpt_names = []
ckpt_names.append("./models/VGG16_0.0001_SVHN_True.pt")
Overlap = 0


transform_test = ToTensor()
batch_size_test = 16
numberOfModels = len(ckpt_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The Device is ', device)
test_set = DATA(root='./data', split='test', download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test)


for epoch in range(numberOfModels):
    PATH = ckpt_names[epoch]
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    network = Network().to(device)
    network.load_state_dict(checkpoint['model_state_dict'])

    total_num_correct = 0
    batch_count = 0
    network.eval()

    with torch.no_grad():
        for batch in loader_test:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            [kasami_preds, preds] = network(images)
            num_correct_batch = preds.argmax(dim=1).eq(labels).sum().item()
            total_num_correct += num_correct_batch

    len_test = len(test_set)
    test_accuarcy = total_num_correct / len_test

    print("Test Accuarcy of Single-Label", PATH, "is : ", test_accuarcy * 100, "%")
    TestType = 'Single-Label'
    Accuracy = test_accuarcy * 100

for epoch in range(numberOfModels):

    PATH = ckpt_names[epoch]
    print(PATH)
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    network = Network().to(device)
    network.load_state_dict(checkpoint['model_state_dict'])
    loss_function_cross = nn.CrossEntropyLoss()

    num_images = 2
    num_correct = 0
    batch_count = 0

    network.eval()
    with torch.no_grad():
        for batch in loader_test:
            batch_count += 1
            rand_indx = np.arange(batch_size_test)
            np.random.shuffle(rand_indx)
            images_1 = batch[0]
            images_2 = images_1[rand_indx]
            images = torch.cat([images_1, images_2], dim=2)
            labels = batch[1]
            labels_rand = labels[rand_indx]
            labels_1 = torch.stack([labels, labels], dim=0)
            labels_2 = torch.stack([labels_rand, labels_rand], dim=0)

            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            images = images.to(device)

            [kasami_preds, preds] = network(images)
            top_k = torch.topk(preds, num_images)
            preds_labels = top_k[1]
            preds_labels = torch.transpose(preds_labels, 0, 1)

            num_correct_now = preds_labels.eq(labels_1).sum().item()
            num_correct += num_correct_now

            num_correct_now = preds_labels.eq(labels_2).sum().item()
            num_correct += num_correct_now

            accuracy = num_correct / (num_images * batch_size_test * batch_count)

        print("Test Accuarcy of Multi-Label", PATH, "is : ", accuracy * 100, "% no Overlap")
        TestType = 'Multi-Label-no-Overlap'
        Accuracy = accuracy * 100


for i in range(0, 11):
    for epoch in range(numberOfModels):
        PATH = ckpt_names[epoch]
        checkpoint = torch.load(PATH, map_location=torch.device(device))
        network = Network().to(device)
        network.load_state_dict(checkpoint['model_state_dict'])
        loss_function_cross = nn.CrossEntropyLoss()

        num_images = 2
        num_correct = 0
        batch_count = 0
        network.eval()
        with torch.no_grad():
            for batch in loader_test:
                batch_count += 1
                rand_indx = np.arange(batch_size_test)
                np.random.shuffle(rand_indx)
                images_1 = batch[0]
                images_2 = images_1[rand_indx]

                images = mergimages(images_1, images_2, i / 10)

                labels = batch[1]
                labels_rand = labels[rand_indx]
                labels_1 = torch.stack([labels, labels], dim=0)
                labels_2 = torch.stack([labels_rand, labels_rand], dim=0)

                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                images = images.to(device)

                [kasami_preds, preds] = network(images)
                top_k = torch.topk(preds, num_images)
                preds_labels = top_k[1]
                preds_labels = torch.transpose(preds_labels, 0, 1)

                num_correct_now = preds_labels.eq(labels_1).sum().item()
                num_correct += num_correct_now

                num_correct_now = preds_labels.eq(labels_2).sum().item()
                num_correct += num_correct_now

                accuracy = num_correct / (num_images * batch_size_test * batch_count)
            print("Test Accuarcy of Multi-Label of", PATH, "is ", accuracy * 100, "%", "Overlap: ", i * 10, "%")
            TestType = 'Multi-Label'
            Overlap = i * 10
            Accuracy = accuracy * 100
