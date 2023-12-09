import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from PIL import Image

from torch.utils.data import Dataset

class cmnist_dataset(Dataset):
  def __init__(self, train,  transform = None):


        self.transform = transform
        #self.train_images =
        #self.train_labels =

  def __getitem__(self, index):
      return r_mnist[index],labels[index]

  def __len__(self):
      return len(r_mnist)

#dataset = MyDataset()

#trainloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True, num_workers = 2)

#datasets.MNIST('datasets/mnist/MNIST', train=True, download=True, transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
