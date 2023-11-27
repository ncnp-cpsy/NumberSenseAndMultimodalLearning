import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import torch
from torch import optim
from torch import nn
from sklearn.metrics import accuracy_score

import torch.nn.functional as F

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='smnist', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=5, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=0, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=2, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# args
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
# load args from disk if pretrained model path is given
args.pre_trained = '../experiments/smnist/best'
pretrained_path = args.pre_trained
args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

if not args.experiment:
    args.experiment = model.modelName

# preparation for training
train_loader, test_loader, abtest_loader = model.getDataLoaders(args.batch_size, device=device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 20)
        self.fc2 = nn.Linear(20,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


def save_image(array, name):
    res = array.cpu().detach().numpy().T
    tar = (res * 255).astype(np.uint8)
    if tar.shape[2] == 1:
        tar = tar[:,:,0]
    pil_image = Image.fromarray(tar) 
    pil_image.save(name)

if __name__ == '__main__':
    
    agg = defaultdict(list)
    model.eval()
    learning_rate = 0.001
    network = Net().to(device)
    network.train()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    
    #訓練 
    for k in range(5):
        for i, dataT in enumerate(train_loader):
            data,target = unpack_data(dataT, device=device, require_label = True)
            output = network(data)
            loss = F.nll_loss(output, target)
            print(loss)
            loss.backward()
            optimizer.step()

    #テスト : 普通ので
    network.eval()

    answer_all = []
    target_all = []
    for i, dataT in enumerate(test_loader):
        data,target = unpack_data(dataT, device=device, require_label = True)
        save_image(data[0], 'check_test.png')
        output = network(data)
        answer = torch.argmax(output, dim=1).cpu().detach().numpy()
        answer_all += list(answer)
        target_all += list(target.cpu().detach().numpy())
    answer_all = np.array(answer_all)
    target_all = np.array(target_all)

    
    accuracy = accuracy_score(answer_all, target_all)
    print('正常ので', accuracy)

    #テスト : 異常で
    answer_all = []
    target_all = []
    for i, dataT in enumerate(abtest_loader):
        data,target = unpack_data(dataT, device=device, require_label = True)
        save_image(data[0], 'check_abtest.png')
        output = network(data)
        answer = torch.argmax(output, dim=1).cpu().detach().numpy()
        answer_all += list(answer)
        target_all += list(target.cpu().detach().numpy())
    answer_all = np.array(answer_all)
    target_all = np.array(target_all)

    accuracy = accuracy_score(answer_all, target_all)
    print('異常ので', accuracy)
   
