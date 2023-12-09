import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from PIL import Image
import create_oscn as co
import time
import collections
from collections import Counter


mnist_train = datasets.MNIST('./datasets/mnist', train=True, download=True)
mnist_train = (mnist_train.data[ mnist_train.targets != 0], mnist_train.targets[ mnist_train.targets != 0])

mnist_test = datasets.MNIST('./datasets/mnist', train=False, download=True)
mnist_test = (mnist_test.data[mnist_test.targets!=0], mnist_test.targets[mnist_test.targets!= 0])


def make_cmnist(images, color):
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
      if tar == i:
        images[torch.tensor(range(len(images))), i, :, :] *= 0

  return (images.float() / 255.)

def execute(phase,n_given ):
    n = n_given

    if phase == 'train':
        images = mnist_train[0]
        labels = mnist_train[1]
    else :
        images = mnist_test[0]
        labels = mnist_test[1]

    r_images = make_cmnist(images[:n], 'r')
    g_images = make_cmnist(images[:n], 'g')
    b_images = make_cmnist(images[:n], 'b')
    #w_images = make_cmnist(images[:n], 'w')
    train_all = torch.cat([r_images, g_images,  b_images],dim = 0)

    train_labels = labels[:n]
    labels_all = torch.cat([train_labels, train_labels,  train_labels], dim = 0)

    #うまく行ったか確認
    idx = np.random.randint(0, n * 3 - 1)
    cs = ['r','g','b']
    tar = (train_all[idx] * 255).int()
    pil_image = Image.fromarray(tar.numpy().astype(np.uint8).T)
    pil_image.save('generated_images/test_color_' + str(cs[int(idx / n)]) + str(labels_all[idx].item()) +'.jpg')

    return train_all,labels_all


cmnist_test_images, cmnist_test_labels = execute('test', 1000)
print(cmnist_test_images.shape, cmnist_test_labels.shape)
torch.save(cmnist_test_images, '../data/abn_cmnist_test_images.pt')
torch.save(cmnist_test_labels, '../data/abn_cmnist_test_labels.pt')

cs = ['r','g','b']
figures = ['square', 'triangle', 'juji']

#oscnを対応させる
def make_oscn(labels):
  img_all = []
  labels_all = []
  colors = [(0, 255 ,255),(255, 0, 255), (255, 255, 0)]
  for i in range(len(labels)):
    figind = np.random.randint(0, 3)
    img = co.create_images(labels[i].item(), figures[figind], colors[int(i /(len(labels)/3))] )
    img_all.append(img.T)
    labels_all.append(cs[int(i /(len(labels)/3))]+str(labels[i].item())+str(figures[figind][0]) )
  print(Counter(labels_all))
  return torch.FloatTensor(img_all), np.array(labels_all, dtype=object)


oscn_test_images, oscn_test_labels = make_oscn(cmnist_test_labels)
print(oscn_test_images.shape, oscn_test_labels.shape)
torch.save(oscn_test_images, '../data/abn_oscn_test_images.pt')
np.save('../data/abn_oscn_test_labels', oscn_test_labels)
