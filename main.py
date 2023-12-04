import os
import sys
import datetime
from collections import defaultdict
from pathlib import Path
import json
import warnings
import pprint
import numpy as np
from numba.core.errors import NumbaPerformanceWarning
import torch

from src.utils import Logger, Timer, save_model, save_vars, unpack_data
import src.models as models
from src.runner import run_train
from src.analyse import analyse
from src.synthesize import synthesize
from src.config import (
    config_trainer_vae_cmnist,
    config_trainer_vae_oscn,
    config_trainer_mmvae_cmnist_oscn,
    config_analyzer_vae_cmnist,
    config_analyzer_vae_oscn,
    config_analyzer_mmvae_cmnist_oscn,
    config_classifier_cmnist,
    config_classifier_oscn,
    config_synthesizer,
)

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True

def update_args(args):
    # load args from disk if pretrained model path is given
    if 'pretrained_path' in dir(args) and args.pretrained_path != '':
        args_tmp = args
        args = torch.load(args.pretrained_path + '/args.rar')
        # No update parameters
        args.run_type = args_tmp.run_type
        args.run_id = args_tmp.run_id
        args.pretrained_path = args_tmp.pretrained_path
        # args.output_dir = args_tmp.output_dir
        # args.no_cuda = args_tmp.no_cuda
        print("args are updated using pretrained_path:\n", args)
    else:
        args.pretrained_path = ''

    # default arguments
    if 'looser' in dir(args):
        args.looser = False
    if 'K' in dir(args):
        args.K = 20
    if 'learn_prior' in dir(args):
        args.learn_prior = False
    if 'llik_scaling' in dir(args):
        args.llik_scaling = 0.0
    if 'logp'in dir(args):
        args.logp = False

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

    # output and log directory
    if args.run_type in ['train', 'analyse']:
        output_dir = os.path.join(run_dir, args.run_type)
        log_path = '{}/{}.log'.format(run_dir, args.run_type)
    elif args.run_type in ['classify']:
        output_dir = os.path.join(run_dir, 'train')
        log_path = '{}/{}.log'.format(run_dir, 'train')
    elif args.run_type in ['synthesize']:
        output_dir = os.path.join(experiment_dir, 'synthesize')
        log_path = '{}/{}.log'.format(experiment_dir, args.run_type)
    else:
        Exception
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(log_path)

    args.run_dir = run_dir
    args.output_dir = output_dir
    return args


def main(args):
    print('Arguments (initial):')
    try:
        pprint.pprint(args)
    except Exception as e:
        print(args)

    if 'seed' in dir(args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args = update_args(args)

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
    if args.pretrained_path == '':
        with open('{}/args.json'.format(args.run_dir), 'w') as fp:
            json.dump(args.__dict__, fp)
        torch.save(args, '{}/args.rar'.format(args.run_dir))
    print('Run ID:\n', args.run_id)
    print('Run Directory:\n', args.run_dir)
    print('Arguments (after settings):')
    try:
        pprint.pprint(args)
    except Exception as e:
        print(args)

    # main
    if args.run_type == 'train':
        run_train(args=args, run_dir=args.run_dir)
    elif args.run_type == 'classify':
        run_train(args=args_classifier, run_dir=args.run_dir)
    elif args.run_type == 'analyse':
        analyse(
            args=args,
            args_classifier_cmnist=args_classifier_cmnist,
            args_classifier_oscn=args_classifier_oscn,
        )
    elif args.run_type == 'synthesize':
        synthesize(
            args=args
        )
    else:
        Exception
    return

def run_all():
    seed_initial, seed_end = 0, 3
    run_ids_dict = defaultdict(list)
    execute_train = True
    experiment_name = config_trainer_vae_cmnist.experiment

    # Train of classifier
    args = config_classifier_cmnist
    args.pretrained_path = ''
    if execute_train:
        main(args=args)

    args = config_classifier_oscn
    args.pretrained_path = ''
    if execute_train:
        main(args=args)

    # Train of VAE and MMVAE
    for seed in range(seed_initial, seed_end):
        args = config_trainer_vae_cmnist
        args.seed = seed
        args.run_id = 'vae_cmnist_seed_' + str(seed)
        run_ids_dict['VAE_CMNIST'].append(args.run_id)
        if execute_train:
            main(args=args)

        args = config_trainer_vae_oscn
        args.seed = seed
        args.run_id = 'vae_oscn_seed_' + str(seed)
        run_ids_dict['VAE_OSCN'].append(args.run_id)
        if execute_train:
            main(args=args)

        args = config_trainer_mmvae_cmnist_oscn
        args.seed = seed
        args.run_id = 'mmvae_cmnist_oscn_seed_' + str(seed)
        run_ids_dict['MMVAE_CMNIST_OSCN'].append(args.run_id)
        if execute_train:
            main(args=args)

    # Analyse results
    for model_name in run_ids_dict.keys():
        for run_id in run_ids_dict[model_name]:
            if model_name == 'VAE_OSCN':
                args = config_analyzer_vae_oscn
            elif model_name == 'VAE_CMNIST':
                args = config_analyzer_vae_cmnist
            elif model_name == 'MMVAE_CMNIST_OSCN':
                args = config_analyzer_mmvae_cmnist_oscn
            else:
                Exception
            args.run_id = run_id
            args.pretrained_path = os.path.join(
                './rslt', experiment_name, model_name, run_id)
            main(args=args)

    # Synthesize
    args = config_synthesizer
    args.pretrained_path = os.path.join(
        './rslt', experiment_name, model_name, run_id) # dummy path
    main(args=args)

    return

def test_train_classifier():
    args = config_classifier_cmnist
    args.pretrained_path = ''
    main(args=args)

    args = config_classifier_oscn
    args.pretrained_path = ''
    main(args=args)
    return

def test_train():
    args = config_trainer_vae_cmnist
    args.run_id = 'test_vae_cmnist'
    main(args=args)

    args = config_trainer_vae_oscn
    args.run_id = 'test_vae_oscn'
    main(args=args)

    args = config_trainer_mmvae_cmnist_oscn
    args.run_id = 'test_mmvae_cmnist_oscn'
    main(args=args)
    return

def test_analyse():
    experiment_name = config_trainer_vae_oscn.experiment

    # # CMNIST
    # args = config_analyzer_vae_cmnist
    # model_name = 'VAE_CMNIST'
    # args.run_id = 'test_vae_cmnist'
    # args.pretrained_path = os.path.join(
    #     './rslt', experiment_name, model_name, args.run_id)
    # main(args=args)

    # # OSCN
    # args = config_analyzer_vae_oscn
    # model_name = 'VAE_OSCN'
    # args.run_id = 'test_vae_oscn'
    # args.pretrained_path = os.path.join(
    #     './rslt', experiment_name, model_name, args.run_id)
    # main(args=args)
    # args = config_analyzer_vae_oscn

    # MMVAE
    args = config_analyzer_mmvae_cmnist_oscn
    model_name = 'MMVAE_CMNIST_OSCN'
    args.run_id = 'test_mmvae_cmnist_oscn'
    args.pretrained_path = os.path.join(
        './rslt', experiment_name, model_name, args.run_id)
    main(args=args)
    return

def test_train_loop():
    run_ids_dict = defaultdict(list)
    execute_train = True
    for seed in range(3, 5):
        args = config_trainer_vae_cmnist
        args.seed = seed
        args.run_id = 'vae_cmnist_seed_' + str(seed)
        run_ids_dict['VAE_CMNIST'].append(args.run_id)
        if execute_train:
            main(args=args)

        args = config_trainer_vae_oscn
        args.seed = seed
        args.run_id = 'vae_oscn_seed_' + str(seed)
        run_ids_dict['VAE_OSCN'].append(args.run_id)
        if execute_train:
            main(args=args)

        # MMVAE
        args = config_trainer_mmvae_cmnist_oscn
        args.seed = seed
        args.run_id = 'mmvae_cmnist_oscn_seed_' + str(seed)
        run_ids_dict['MMVAE_CMNIST_OSCN'].append(args.run_id)
        if execute_train:
            main(args=args)
    return run_ids_dict

def test_analyse_loop(run_ids_dict):
    experiment_name = config_trainer_vae_cmnist.experiment
    for model_name in run_ids_dict.keys():
        for run_id in run_ids_dict[model_name]:
            if model_name == 'VAE_OSCN':
                args = config_analyzer_vae_oscn
            elif model_name == 'VAE_CMNIST':
                args = config_analyzer_vae_cmnist
            elif model_name == 'MMVAE_CMNIST_OSCN':
                args = config_analyzer_mmvae_cmnist_oscn
            else:
                Exception
            args.run_id = run_id
            args.pretrained_path = os.path.join(
                './rslt', experiment_name, model_name, run_id)
            main(args=args)

def test_synthesize():
    args = config_synthesizer
    experiment_name = config_trainer_vae_cmnist.experiment
    model_name = 'VAE_OSCN'
    run_id = 'test_vae_oscn'
    args.pretrained_path = os.path.join(
        './rslt', experiment_name, model_name, run_id)
    main(args=args)

def test_pipeline():
    test_train_classifier()
    run_ids_dict = test_train_loop()
    test_analyse_loop(run_ids_dict=run_ids_dict)
    args.run_ids_dict = run_ids_dict
    main(args=config_synthesizer)

def test_run_all():
    run_all()

def test():
    # test_train_classifier()
    # test_train()
    # test_analyse()
    # test_synthesize()
    # test_pipeline()
    test_run_all()

if __name__ == '__main__':
    test()
    # main(args=config_trainer_vae_oscn)
    # run_all()
