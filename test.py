import numpy as np
import torch
from tqdm.auto import tqdm

def merge_images(x, y, overlap):
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


def apply_merge_to_batch(batch, overlap, batch_size, device):
    rand_indx = np.arange(batch_size)
    np.random.shuffle(rand_indx)
    images_1 = batch[0]
    images_2 = images_1[rand_indx]
    images = merge_images(images_1, images_2, overlap)
    labels = batch[1]
    labels_rand = labels[rand_indx]
    labels_1 = torch.stack([labels, labels], dim=0)
    labels_2 = torch.stack([labels_rand, labels_rand], dim=0)
    images = images.to(device)
    labels_1 = labels_1.to(device)
    labels_2 = labels_2.to(device)
    return images, labels_1, labels_2


def get_num_correct_for_merge(preds, labels_1, labels_2, num_images=2):
    top_k = torch.topk(preds, num_images)
    preds_labels = top_k[1]
    preds_labels = torch.transpose(preds_labels, 0, 1)
    num_correct = preds_labels.eq(labels_1).sum().item()
    num_correct += preds_labels.eq(labels_2).sum().item()
    return num_correct


def evaluate_model(network, loader_test, device):
    batch_size = loader_test.batch_size
    batch_count = len(loader_test)
    total_images = batch_count * batch_size
    total_num_correct = 0
    num_correct_batch = 0
    network.eval()
    with torch.no_grad():
        eval_bar = tqdm(loader_test, desc=f'Evaluating Batch Correct: {num_correct_batch}/{batch_size} Total: {total_num_correct}/{total_images}')
        for batch in eval_bar:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            [kasami_preds, preds] = network(images)
            num_correct_batch = preds.argmax(dim=1).eq(labels).sum().item()
            total_num_correct += num_correct_batch
            eval_bar.set_description(f'Evaluating Batch Correct: {num_correct_batch}/{batch_size} Total: {total_num_correct}/{total_images}')
            eval_bar.update()
    test_accuracy = total_num_correct / (total_images)
    return test_accuracy


def evaluate_model_with_overlap(network, loader_test, device, overlap, num_images=2):
    batch_size = loader_test.batch_size
    batch_count = len(loader_test)
    total_num_images = batch_count * batch_size * num_images
    total_num_correct = 0
    num_correct_batch = 0
    network.eval()
    with torch.no_grad():
        eval_bar = tqdm(loader_test, desc=f'Evaluating Batch Correct: {num_correct_batch}/{batch_size*num_images} Total: {total_num_correct}/{total_num_images}')
        for batch in loader_test:
            batch_size = batch[0].shape[0]
            images, labels_1, labels_2 = apply_merge_to_batch(batch, overlap, batch_size, device)
            [kasami_preds, preds] = network(images)
            num_correct_batch = get_num_correct_for_merge(preds, labels_1, labels_2)
            total_num_correct += num_correct_batch
            eval_bar.set_description(f'Evaluating Batch Correct: {num_correct_batch}/{batch_size*num_images} Total: {total_num_correct}/{total_num_images}')
            eval_bar.update()
    test_accuracy = total_num_correct / (total_num_images)
    return test_accuracy

        
        
