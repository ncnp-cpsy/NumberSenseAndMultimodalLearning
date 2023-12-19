import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from PIL import Image
import time
import collections
from collections import Counter
from torch.utils.data import Dataset
from torchnet.dataset import TensorDataset, ResampleDataset

import cv2
import numpy as np
from random import randint
import random
from PIL import Image
import itertools

height = 32
width = height

size = 6
pos_all = [
    (5, 5 ),
    (5, 15),
    (5, 25),
    (15, 5 ),
    (15, 15),
    (15, 25),
    (25, 5 ),
    (25, 15),
    (25, 25),
]

""" size = 4
nums = [10 * i for i in range(1, int(height/size) - 4)]
pos_all = list(itertools.product(nums, repeat=2))
print(pos_all) """

def is_hit(ver1, ver2, fig_type):
    if fig_type == "square" or fig_type == "juji" or fig_type == "triangle":
        if ( abs(ver2[0] - ver1[0]) < size ) and ( abs(ver2[1] - ver1[1]) < size ) :
            return True
    if fig_type == "circle" :
        if ( (ver2[0] - ver1[0]) ** 2.0 + (ver2[1] - ver1[1]) ** 2.0 < (size) ** 2.0 ):
            return True
    return False

def are_hit(ver, dones, fig_type):
    for i in range(len(dones)):
        if is_hit(ver, dones[i], fig_type):
            return True
    return False

def draw(ver, size, fig_type, img, color):
    colors = [(0, 255 ,255),(255, 0, 255), (255, 255, 0), (255,255,255)]
    color = colors[np.random.randint(0, len(colors))]
    if fig_type == "square":
        for i in range(ver[0] - int(size/2), ver[0] + int(size/2)):
            for j in range(ver[1] - int(size/2), ver[1] + int(size/2)):
                #(img[i,j,0], img[i,j,1], img[i,j,2]) =  color
                (img[j, i, 0], img[j, i, 1], img[j, i, 2]) =  color
    elif fig_type == "circle":
        cv2.circle(img, (ver[0],ver[1]), int(size / 2),  color, thickness= -1 )
    elif fig_type == "juji":
        cv2.drawMarker(img, (ver[0], ver[1]),  color , markerType=cv2.MARKER_CROSS, markerSize=size ) 
    elif fig_type == 'triangle':
        alpha = (size * (3 ** 0.5))/2
        #triangle_cnt = np.array( [(ver[0], ver[1]), (ver[0] + size, ver[1]), ( int(ver[0] + size /2), int (ver[1] - size * (3 ** 0.5)/2 ) )] )
        triangle_cnt = np.array( [(ver[0] - int(size/2) , ver[1] + int(size /(2 * (3 ** 0.5))) + 1 ), (ver[0] + int(size/2), ver[1] + int(size /(2 * (3 ** 0.5))) +1 ), ( ver[0] , int (ver[1] - size * (3 ** 0.5)/2 ) + 2 )]  )
        cv2.drawContours(img, [triangle_cnt], 0,  color, -1)



def create_images(num, fig_type, color, randomize_fig = False):
    img = np.zeros((height, width, 3), np.uint8)
    dones = []

    if num == 0:
        return img / 255.0

    #まず、どのブロックに配置するか決める
    block_idx = random.sample(range(len(pos_all)),num)

    for id in block_idx:
        pt = pos_all[id]
        if randomize_fig:
            figures = ['square', 'triangle', 'juji']   
            figind =  np.random.randint(0, len(figures))
            draw(pt, size, figures[figind], img, color)
        else :
            draw(pt, size, fig_type, img, color)
    return img / 255.0



def save_image(img, fig_type, num):
    cv2.imwrite('images/' + fig_type + str(num) + '.png', img)




mnist_train = datasets.MNIST('../cmniste_test/datasets/mnist', train=True, download=True)
mnist_train = (mnist_train.data[ mnist_train.targets != 0], mnist_train.targets[ mnist_train.targets != 0])

mnist_test = datasets.MNIST('../cmniste_test/datasets/mnist', train=False, download=True)
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
    w_images = make_cmnist(images[:n], 'w')
    train_all = torch.cat([r_images, g_images,  b_images, w_images],dim = 0)

    train_labels = labels[:n]
    labels_all = torch.cat([train_labels, train_labels,  train_labels,  train_labels], dim = 0)

    #うまく行ったか確認
    """ idx = np.random.randint(0, n * 3 - 1)
    cs = ['r','g','b']
    tar = (train_all[idx] * 255).int()
    pil_image = Image.fromarray(tar.numpy().astype(np.uint8).T)
    pil_image.save('generated_images/test_color_' + str(cs[int(idx / n)]) + str(labels_all[idx].item()) +'.jpg') """

    return train_all,labels_all


cmnist_test_images, cmnist_test_labels = execute('test', 1000)
print(cmnist_test_images.shape, cmnist_test_labels.shape)



#oscnを対応させる
def make_oscn(labels):
  figures = ['square', 'triangle', 'juji']   
  img_all = []
  labels_all = []
  cs = ['r','g','b', 'w']
  #colors = [(255,0, 0),(0,255, 0), (0, 0, 255)]
  colors = [(0, 255 ,255),(255, 0, 255), (255, 255, 0), (255,255,255)] #異常な色
  for i in range(len(labels)):
    figind = np.random.randint(0, len(figures))
    #img = create_images(labels[i].item(), figures[figind], colors[int(i /(len(labels)/len(colors)))] , randomize_fig = True)
    img = create_images(np.random.randint(0, len(pos_all)), figures[figind], colors[int(i /(len(labels)/len(colors)))] , randomize_fig = True)
    img_all.append(img.T)
    
    labels_all.append(cs[int(i /(len(labels)/len(colors)))]+str(labels[i].item())+str(figures[figind][0]) )
  #print(Counter(labels_all))
  return torch.FloatTensor(img_all), np.array(labels_all, dtype=object)


oscn_test_images, oscn_test_labels = make_oscn(cmnist_test_labels)
print(oscn_test_images.shape, oscn_test_labels.shape)


## write code here
class cmnist_dataset(Dataset):
  def __init__(self):
        self.images = cmnist_test_images
        self.labels = cmnist_test_labels
  def __getitem__(self, index):
      return self.images[index], self.labels[index]
  def __len__(self):
      return len(self.images)

cmnist_dataset = cmnist_dataset()

class oscn_dataset(Dataset):
  def __init__(self):
        self.images = oscn_test_images
        self.labels = oscn_test_labels
  def __getitem__(self, index):
      return self.images[index], self.labels[index]
  def __len__(self):
      return len(self.images)

oscn_dataset = oscn_dataset()

batch_size = 128
shuffle = True



cmnist_abn_loader = torch.utils.data.DataLoader(cmnist_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
oscn_abn_loader = torch.utils.data.DataLoader(oscn_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)

cmnist_oscn_abn_loader = torch.utils.data.DataLoader(TensorDataset([cmnist_dataset, oscn_dataset]), batch_size = batch_size, shuffle = shuffle, num_workers = 2)