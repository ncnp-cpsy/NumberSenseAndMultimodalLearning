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
from torchvision.utils import save_image, make_grid



mnist_train = datasets.MNIST('./datasets/mnist', train=True, download=True)
mnist_train = (mnist_train.data[ mnist_train.targets != 0], mnist_train.targets[ mnist_train.targets != 0])

mnist_test = datasets.MNIST('./datasets/mnist', train=False, download=True)
mnist_test = (mnist_test.data[mnist_test.targets!=0], mnist_test.targets[mnist_test.targets!= 0])

#rng_state = np.random.get_state()
#np.random.shuffle(images[0].numpy())
#np.random.set_state(rng_state)
#np.random.shuffle(images[1].numpy())

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
      if tar != i:
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
    w_images = make_cmnist(images[:n], 'w')
    train_all = torch.cat([r_images, g_images, b_images, w_images],dim = 0)
    #train_all = torch.cat([w_images],dim = 0)

    train_labels = labels[:n]
    labels_all = torch.cat([train_labels, train_labels, train_labels, train_labels], dim = 0)
    #labels_all = torch.cat([train_labels], dim = 0)

    return train_all,labels_all


cmnist_train_images, cmnist_train_labels = execute('train',30000)
print(cmnist_train_images.shape, cmnist_train_labels.shape)
torch.save(cmnist_train_images, '../data/cmnist_train_images.pt')
torch.save(cmnist_train_labels, '../data/cmnist_train_labels.pt')

cmnist_test_images, cmnist_test_labels = execute('test', 1000)
print(cmnist_test_images.shape, cmnist_test_labels.shape)
torch.save(cmnist_test_images, '../data/cmnist_test_images.pt')
torch.save(cmnist_test_labels, '../data/cmnist_test_labels.pt')

cs = ['r','g','b','w']
#cs = ['w']
figures = ['square', 'triangle', 'juji']
#figures = ['square']
include_zero = False

#oscnを対応させる
def make_oscn(labels):
  img_all = []
  labels_all = []
  colors = [(255, 0 ,0),(0, 255,0), (0, 0, 255), (255, 255, 255)]
  #colors = [(255, 255, 255)]
  for i in range(len(labels)):
    figind = np.random.randint(0, len(figures))


    if include_zero:
      judge = np.random.randint(0,10) == 0
      if judge:
        img = co.create_images(0, figures[figind], colors[int(i /(len(labels)/len(colors)))] )
      else:
        img = co.create_images(labels[i].item(), figures[figind], colors[int(i /(len(labels)/len(colors)))] )
    else:
      img = co.create_images(labels[i].item(), figures[figind], colors[int(i /(len(labels)/len(colors)))] )

    img_all.append(img.T)
    if include_zero:
      if judge:
        labels_all.append(cs[int(i /(len(labels)/len(colors)))]+str(0)+str(figures[figind][0]) )
      else:
        labels_all.append(cs[int(i /(len(labels)/len(colors)))]+str(labels[i].item())+str(figures[figind][0]) )
    else:
      labels_all.append(cs[int(i /(len(labels)/len(colors)))]+str(labels[i].item())+str(figures[figind][0]) )
  print(Counter(labels_all))
  return torch.FloatTensor(img_all), np.array(labels_all, dtype=object)

oscn_train_images, oscn_train_labels = make_oscn(cmnist_train_labels)
print(oscn_train_images.shape, oscn_train_labels.shape)
torch.save(oscn_train_images, '../data/oscn_train_images.pt')
np.save( '../data/oscn_train_labels', oscn_train_labels)


oscn_test_images, oscn_test_labels = make_oscn(cmnist_test_labels)
print(oscn_test_images.shape, oscn_test_labels.shape)
torch.save(oscn_test_images, '../data/oscn_test_images.pt')
np.save('../data/oscn_test_labels', oscn_test_labels)

#画像の確認

#うまく行ったか確認
inds = [np.random.randint(0, len(cmnist_train_images)) for i in range(100)]
save_image(cmnist_train_images[inds], 'generated_images/test_cmnist.jpg')
save_image(oscn_train_images[inds], 'generated_images/test_oscn.jpg')


""" for i in range(10):
  ind = np.random.randint(0, len(cmnist_train_images))
  tar = (cmnist_train_images[ind] * 255).int()
  pil_image = Image.fromarray(tar.numpy().astype(np.uint8).T)
  pil_image.save('generated_images/test_cmnist' + str(i) + '.jpg')

  tar = np.array((oscn_train_images[ind]* 255)).T.astype(np.uint8)
  pil_image = Image.fromarray(tar)
  pil_image.save('generated_images/test_oscn' + str(i) + '.png') """