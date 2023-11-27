import argparse
import sys
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import optim

import models
from utils import Logger, Timer, save_model, save_vars, unpack_data
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# args
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

for key, param in model.state_dict().items():
    print(key, param.shape)
    print(param)
