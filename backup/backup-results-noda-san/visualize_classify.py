import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import torch
from torch import optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
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
pretrained_path = ""
if args.pre_trained:
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
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))
t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')


def train(epoch, agg):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


if __name__ == '__main__':
    if True:
        agg = defaultdict(list)
        model.eval()

        label_all = []
        latent_all = []

        # cmnist か oscn か、ということ。0ならcmnist, 1ならoscn 
        target_modality = 0
        # その中で 0 : 色、　1 : 数、 2: 形
        n_modality = 0

        require_ps = False
        require_3d = True

        answer_all = []
        target_all = []
        for i, dataT in enumerate(train_loader):
            data, label = unpack_data(dataT, device=device, require_label = True, target = target_modality)

            # abnormalize
            data[0] = data[0][torch.randperm(data[0].size()[0])]
            data[1] = data[1][torch.randperm(data[1].size()[0])]

            qz_xs, px_zs, zss, cond_probs = model([data[0], data[1]], K = 1)
            print(cond_probs.shape)
            answer = torch.argmax(cond_probs, dim=1).cpu().detach().numpy()
            answer_all += list(answer)
            target_all += list(label.cpu().detach().numpy())

            if i == 5:
                break

        answer_all = np.array(answer_all)
        target_all = np.array(target_all)

        accuracy = accuracy_score(answer_all, target_all)
        print('正常ので', accuracy)
