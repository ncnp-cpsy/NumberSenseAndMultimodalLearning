from torchvision import datasets, transforms
from torch import utils
import torch
import vae
from vae import VAE

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)


import numpy as np
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

model = VAE(20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for i in range(20):
  losses = []
  for x, t in dataloader_train:
      
      x = x.to(device)
      model.zero_grad()
      y = model(x)

      
      
      loss = model.loss(x)
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  if i == 0:
      print('xshape : ',x.shape)
      print('yshape:', y[0].shape, y[1].shape)
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from random import random

colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]



def visualize_zs(zs, labels):
  plt.figure(figsize=(10,10))
  points = TSNE(n_components=2, random_state=0).fit_transform(zs)
  for p, l in zip(points, labels):
    plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l])
  plt.savefig('latest.png')

model.eval()
zs = []
for x, t in dataloader_valid:
    x = x.to(device)
    t = t.to(device)
    # generate from x
    y, z = model(x)
    z = z.cpu()
    t = t.cpu()
    print('xshape shin :', x.shape)
    print('zshape shin :', z.shape)
    visualize_zs(z.detach().numpy(), t.cpu().detach().numpy())
    break
