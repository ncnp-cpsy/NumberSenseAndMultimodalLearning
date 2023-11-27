import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from PIL import Image
import time
import collections
from collections import Counter


mnist_train = datasets.MNIST('./datasets/mnist', train=True, download=True)
mnist_train = (mnist_train.data, mnist_train.targets)

mnist_test = datasets.MNIST('./datasets/mnist', train=False, download=True)
mnist_test = (mnist_test.data, mnist_test.targets)

#rng_state = np.random.get_state()
#np.random.shuffle(images[0].numpy())
#np.random.set_state(rng_state)
#np.random.shuffle(images[1].numpy())

cdict = {
  0 : 'w',
  1 : 'r',
  2 : 'g',
  3 : 'b',
  4 : 'w',
  5 : 'r',
  6 : 'g',
  7 : 'b',
  8 : 'w',
  9 : 'r'
}

abnormal_cdict = {
  0 : 'r',
  1 : 'r',
  2 : 'b',
  3 : 'w',
  4 : 'r',
  5 : 'g',
  6 : 'b',
  7 : 'w',
  8 : 'r',
  9 : 'g'
}

def make_smnist(images, color):
  #images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images, images], dim=1)

  if color != 'w':
    if color == 'r':
      tar = 0
    elif color == 'g':
      tar = 1
    elif color == 'b':
      tar = 2
    for i in range(3):
      if tar != i:
        images[torch.tensor(range(len(images))), i, :, :] *= 0

  return ()

def execute(phase):

    if phase == 'train':
        images = mnist_train[0]
        labels = mnist_train[1]
    else :
        images = mnist_test[0]
        labels = mnist_test[1]

    images = torch.stack([images, images, images], dim=1)

    for i in range(len(images)):
      if phase == 'train' or phase == 'test':
        color = cdict[labels[i].item()]
      elif phase == 'ab_test':
        color = abnormal_cdict[labels[i].item()]

      if color != 'w':
        if color == 'r':
          tar = 0
        elif color == 'g':
          tar = 1
        elif color == 'b':
          tar = 2

        for k in range(3):
          if tar != k:
            images[i, k, :, :] *= 0
      
    images = images.float() / 255.

    for a in range(10):
      #うまく行ったか確認
      idx = np.random.randint(0, len(images))
      cs = ['r','g','w', 'b']
      tar = (images[idx] * 255).int()
      pil_image = Image.fromarray(tar.numpy().astype(np.uint8).T)
      pil_image.save('generated_images/test_' + phase + str(labels[idx].item()) +'.jpg')
      
    return images,labels


smnist_train_images, smnist_train_labels = execute('train')
print(smnist_train_images.shape, smnist_train_labels.shape)
torch.save(smnist_train_images, '../data/smnist_train_images.pt')
torch.save(smnist_train_labels, '../data/smnist_train_labels.pt')

smnist_test_images, smnist_test_labels = execute('test')
print(smnist_test_images.shape, smnist_test_labels.shape)
torch.save(smnist_test_images, '../data/smnist_test_images.pt')
torch.save(smnist_test_labels, '../data/smnist_test_labels.pt')


smnist_abtest_images, smnist_abtest_labels = execute('ab_test')
print(smnist_abtest_images.shape, smnist_abtest_labels.shape)
torch.save(smnist_abtest_images, '../data/smnist_abtest_images.pt')
torch.save(smnist_abtest_labels, '../data/smnist_abtest_labels.pt')
