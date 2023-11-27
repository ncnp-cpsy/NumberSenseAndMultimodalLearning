import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from PIL import Image
import time

import numpy as np
import torch
from torch import optim
# import torchsummary

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data
from torchvision.utils import save_image, make_grid

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg', 'cross'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
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
parser.add_argument('--use-conditional', action='store_true', default=False,
                    help='add conditional term')

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

# load model
model_class = getattr(models, '{}'.format(args.model))
model = model_class(args).to(device)
# torchsummary.summary(model)

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

if not args.experiment:
    args.experiment = model.modelName


# set up run path
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('../experiments/' + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)

if args.model == 'smnist':
    train_loader, test_loader, abtest_loader = model.getDataLoaders(
        args.batch_size, device=device)
else:
    train_loader, test_loader = model.getDataLoaders(
        args.batch_size, device=device)

objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))
t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')


def get_image(data, name):
    data = (data * 255).int()
    data = data.cpu().numpy().astype(np.uint8).T
    pil_image = Image.fromarray(data)
    pil_image.save(name)


def train(epoch, agg):
    """Training models
    """
    model.train()
    b_loss = 0
    start_time = time.time()

    for i, dataT in enumerate(train_loader):
        data, label = unpack_data(
            dataT, device=device, require_label=True)

        #save_image(data[0:3], 'checks/clevr.png')
        """
        for i in range(10,20):
            # get_image(data[0][i], 'cmnist' + str(i) + '.jpg')
            tar = np.array((data[i]* 255).cpu()).T.astype(np.uint8)
            pil_image = Image.fromarray(tar)
            # pil_image.save('generated_images/test_oscn' + str(i) + '.png')
            # pil_image = Image.fromarray(
                data[i].int().cpu().numpy().astype(np.uint8).T)
            pil_image.save('checks/oscn' + str(i) + '.png')
        """
        optimizer.zero_grad()
        loss = objective(
            model,
            data,
            K=args.K,
            conditional=args.use_conditional,
            labels=label,
            device=device)
        loss = -1 * loss
        """
        with open(str(runPath) + '/ratio.log', 'a') as f:
            print(ratio[0], ratio[1], file=f)
        """
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(
                i, loss.item() / args.batch_size))

    end_time = time.time()
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(
        epoch, agg['train_loss'][-1]), " took :", end_time - start_time)


def test(epoch, agg):
    """Testing models
    """
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data, label = unpack_data(
                dataT, device=device, require_label=True)
            model.reconstruct(data, runPath, epoch)
            if i == 0:
                break

            loss = -t_objective(model, data, K=args.K)
            b_loss += loss.item()
            if i == 0:
                # model.reconstruct(data, runPath, epoch)
                print('done!')
                break
                # if not args.no_analytics:
                    # model.analyse(data, runPath, epoch)
    # agg['test_loss'].append(b_loss / len(test_loader.dataset))
    # print('====> Test loss: {:.4f}'.format(agg['test_loss'][-1]))


def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in test_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -t_objective(model, data, K).item()

    marginal_loglik /= len(test_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            model.generate(runPath, epoch)
            test(epoch, agg)
            
        if args.logp:  # compute as tight a marginal likelihood as possible
            estimate_log_marginal(5000)

