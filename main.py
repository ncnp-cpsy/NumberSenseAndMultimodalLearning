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

from src.utils import Logger, Timer, save_model, save_vars, unpack_data
import src.models as models
from src.config import config_classifier_cmnist, config_classifier_oscn
from src.runner import run_train
from src.analyse import analyse


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')

# Experiment Model settings
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--run-type', type=str, default='train', metavar='R',
                    choices=['train', 'classify', 'analyse', 'synthesize'],
                    help='types of run (default: train)')
parser.add_argument('--model', type=str, default='VAE_OSCN', metavar='M',
                    choices=[s.__name__ for s in models.__all__],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg', 'cross'],
                    help='objective to use (default: elbo)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')

# Loss
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')

# Learning
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--use-conditional', action='store_true', default=False,
                    help='add conditional term')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Analyse
# parser.add_argument('--target-modality', type=int, default=0, metavar='M',
#                     help='analysis target of information modality (default: 0)')
# parser.add_argument('--target-property', type=int, default=1, metavar='P',
#                     help='analysis target of information property (default: 1)')
parser.add_argument('--output-dir', type=str, default="./", metavar='D',
                    help='save directory of results (default: latent_image)')


# args
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def main(run_type, args):
    print('Arguments (initial):\n', args)

    # load args from disk if pretrained model path is given
    args.pretrained_path = ""
    if args.pre_trained:
        # temporary
        args_tmp_output_dir = args.output_dir
        args.pretrained_path = args.pre_trained
        # load
        args = torch.load(args.pre_trained + '/args.rar')
        args.output_dir = args_tmp_output_dir
        print("args are updated using pretrained_path:\n", args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.device = torch.device("cuda" if args.cuda else "cpu")
    args.device = "cuda" if args.cuda else "cpu"

    # set up directories
    run_id = datetime.datetime.now().isoformat()
    run_path = None
    if not args.experiment:
        args.experiment = args.model
    experiment_dir = Path('./rslt/' + args.experiment)
    if args.pre_trained:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        run_path = mkdtemp(prefix=run_id, dir=str(experiment_dir))
        sys.stdout = Logger('{}/run.log'.format(run_path))
        with open('{}/args.json'.format(run_path), 'w') as fp:
            json.dump(args.__dict__, fp)
        torch.save(args, '{}/args.rar'.format(run_path))

    # log
    print('Expt:', run_path)
    print('RunID:', run_id)
    print('Arguments (after settings):\n', args)

    # main
    if run_type == 'train':
        run_train(args=args, run_path=run_path)

    elif run_type == 'classify':
        # NOTE: comannd line arguments were not used for classify
        print('Parameters were loaded for classifier')
        if 'Classifier_CMNIST' in args.model:
            args_classifier = config_classifier_cmnist
        elif 'Classifier_OSCN' in args.model:
            args_classifier = config_classifier_oscn
        else:
            Exception
        run_train(args=args_classifier, run_path=run_path)

    elif run_type == 'analyse':
        analyse(
            args=args,
            args_classifier_cmnist=config_classifier_cmnist,
            args_classifier_oscn=config_classifier_oscn,
        )
    elif run_type == 'synthesize':
        synthesize()
    else:
        Exception
    return


if __name__ == '__main__':
    main(run_type=args.run_type, args=args)
