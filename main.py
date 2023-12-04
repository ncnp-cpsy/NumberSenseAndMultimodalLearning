import os
import sys
import datetime
from pathlib import Path
import json
import warnings

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
import torch

from src.utils import Logger, Timer, save_model, save_vars, unpack_data
import src.models as models
from src.runner import run_train
from src.analyse import analyse
from src.config import (
    config_trainer_vae_cmnist,
    config_trainer_vae_oscn,
    config_analyzer_vae_cmnist,
    config_analyzer_vae_oscn,
    config_classifier_cmnist,
    config_classifier_oscn,
)

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True

def main(args):
    print('Arguments (initial):\n', args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load args from disk if pretrained model path is given
    if 'pretrained_path' in dir(args) and args.pretrained_path != '':
        args_tmp = args
        args = torch.load(args.pretrained_path + '/args.rar')
        # No update parameters
        args.run_type = args_tmp.run_type
        args.run_id = args_tmp.run_id
        args.pretrained_path = args_tmp.pretrained_path
        # args.output_dir = args_tmp.output_dir
        print("args are updated using pretrained_path:\n", args)
    else:
        args.pretrained_path = ''

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # args.device = torch.device("cuda" if args.cuda else "cpu")

    # set up directories
    if (not 'experiment' in dir(args)) \
       or ('experiment' in dir(args) and args.experiment == ''):
        args.experiment = 'test'
    if (not 'run_id' in dir(args)) \
       or ('run_id' in dir(args) and args.run_id == ''):
        args.run_id = datetime.datetime.now().isoformat()
    experiment_dir = os.path.join('./rslt/' + args.experiment)
    model_dir = os.path.join(experiment_dir, args.model)
    run_dir = os.path.join(model_dir, args.run_id)
    output_dir = os.path.join(
        run_dir, 'train' if args.run_type == 'classify' else args.run_type)
    # Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    # classifier. comannd line arguments were not used for classify
    print('Parameters were loaded for classifier')
    args_classifier_cmnist = config_classifier_cmnist
    args_classifier_oscn = config_classifier_oscn
    if 'Classifier_CMNIST' in args.model:
        args_classifier = args_classifier_cmnist
    elif 'Classifier_OSCN' in args.model:
        args_classifier = args_classifier_oscn
    else:
        Exception

    # log
    sys.stdout = Logger('{}/{}.log'.format(
        run_dir,
        'train' if args.run_type == 'classify' else args.run_type))
    if args.pretrained_path == '':
        with open('{}/args.json'.format(run_dir), 'w') as fp:
            json.dump(args.__dict__, fp)
        torch.save(args, '{}/args.rar'.format(run_dir))
    print('Expt:\n', experiment_dir)
    print('RunID:\n', args.run_id)
    print('Run_dir\n', run_dir)
    print('Arguments (after settings):\n', args)

    # main
    if args.run_type == 'train':
        run_train(args=args, run_dir=run_dir)
    elif args.run_type == 'classify':
        run_train(args=args_classifier, run_dir=run_dir)
    elif args.run_type == 'analyse':
        analyse(
            args=args,
            args_classifier_cmnist=args_classifier_cmnist,
            args_classifier_oscn=args_classifier_oscn,
        )
    elif args.run_type == 'synthesize':
        synthesize()
    else:
        Exception
    return

def test_train():
    main(args=config_trainer_vae_cmnist)
    main(args=config_trainer_vae_oscn)

def test_train_classifier():
    args = config_classifier_cmnist
    args.pretrained_path = ''
    main(args=args)
    args = config_classifier_cmnist
    args.pretrained_path = ''
    main(args=args)

def test_analyse():
    main(args=config_analyzer_vae_cmnist)
    main(args=config_analyzer_vae_oscn)

def test_pipeline():
    main(args=config_trainer_vae_cmnist)
    main(args=config_trainer_vae_oscn)
    main(args=config_classifier_cmnist)
    main(args=config_classifier_oscn)
    main(args=config_analyzer_vae_cmnist)
    main(args=config_analyzer_vae_oscn)

if __name__ == '__main__':
    # test_train()
    test_analyse()
