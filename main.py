import argparse
import datetime
import sys
import json
from pathlib import Path
import warnings
from tempfile import mkdtemp

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
import torch

from src.runner import run_train
from src.utils import Logger, Timer, save_model, save_vars, unpack_data
import src.models as models

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--run-type', type=str, default='train', metavar='R',
                    choices=['train', 'classify', 'analyze', 'synthesize'],
                    help='types of run (default: train)')
parser.add_argument('--model', type=str, default='VAE_OSCN', metavar='M',
                    choices=[s.__name__ for s in models.__all__],
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


def main(args):
    # load args from disk if pretrained model path is given
    args.pretrained_path = ""
    if args.pre_trained:
        args.pretrained_path = args.pre_trained
        args = torch.load(args.pre_trained + '/args.rar')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.device = torch.device("cuda" if args.cuda else "cpu")
    args.device = "cuda" if args.cuda else "cpu"

    if not args.experiment:
        # args.experiment = model.modelName
        args.experiment = args.model

    # set up run path
    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('./experiments/' + args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_path = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(run_path))

    # log
    print('Expt:', run_path)
    print('RunID:', runId)
    print('Arguments:', args)

    # save args to run
    with open('{}/args.json'.format(run_path), 'w') as fp:
        json.dump(args.__dict__, fp)
    torch.save(args, '{}/args.rar'.format(run_path))

    if args.run_type == 'train':
        run_train(args=args, run_path=run_path)
    elif args.run_type == 'classify':
        classify()
    elif args.run_type == 'analyze':
        analyze()
    elif args.run_type == 'synthesize':
        synthesize()
    else:
        Exception
    return

if __name__ == '__main__':
    main(args=args)
