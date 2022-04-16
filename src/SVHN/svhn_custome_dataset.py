from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pickle
from torch.nn.functional import one_hot
import random

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return  img
        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return image

class SVHN_Dataset(Dataset):
    """SVHN Orginal dataset."""

    def __init__(self, digit_struct_file, root_dir, transform=None):
        """
        Args:
            digit_struct_file (string): Path to the digit struct file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.digit_struct = h5py.File(digit_struct_file, 'r')
        self.root_dir = root_dir
        self.transform = transform
        self.meta_data = self.get_all_boxes()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
      filepath = '%s/%d.png'%(self.root_dir, index+1)
      img = cv2.imread(filepath)
      img = Rescale((256,256)).__call__(img)
      img = ToTensor().__call__(img)
      # print(filepath, img.shape)
      if self.transform:
        img = self.transform(img)
      b_box = self.get_box_data(index)
      labels = self.get_labels(b_box)
      sample = {'image': img, 'labels': labels}
      return sample

    def get_box_data(self, index):
      meta_data = dict()
      meta_data['height'] = []
      meta_data['label'] = []
      meta_data['left'] = []
      meta_data['top'] = []
      meta_data['width'] = []

      def print_attrs(name, obj):
          vals = []
          if obj.shape[0] == 1:
              vals.append(obj[0][0])
          else:
              for k in range(obj.shape[0]):
                  vals.append(int(self.digit_struct[obj[k][0]][0][0]))
          meta_data[name] = vals
          
      box = self.digit_struct['/digitStruct/bbox'][index]
      self.digit_struct[box[0]].visititems(print_attrs)
      return meta_data
    
    def get_all_boxes(self):
      b_boxes = []
      for i in tqdm(range(len(self.digit_struct['/digitStruct/name'])), 'Loading Image Dataset From .mat File'):
        # if i == 100:
        #   break
        b_boxes.append(self.get_box_data(i))
      return b_boxes

    def get_labels(self, bbox):
        
        l = []
        for x in bbox['label'] :
          if x == 10 :
            l.append(0)
          else:
            l.append(x)
        new_labels = torch.tensor(l, dtype=torch.int64)
        labels = one_hot(new_labels,  num_classes=10)
        labels_sum = torch.sum(labels, dim=0)
        # for i,x in enumerate(labels_sum):
        #   if x ==0 :
        #     labels_sum[i] = 0
        #   else:
        #     labels_sum[i] = 1

        return labels_sum

