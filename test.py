import os
import copy
from collections import defaultdict

from main import main, run_all
from src.config_test import (
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

    # CMNIST
    args = config_analyzer_vae_cmnist
    model_name = 'VAE_CMNIST'
    args.run_id = 'test_vae_cmnist'
    args.pretrained_path = os.path.join(
        './rslt', experiment_name, model_name, args.run_id)
    main(args=args)

    # OSCN
    args = config_analyzer_vae_oscn
    model_name = 'VAE_OSCN'
    args.run_id = 'test_vae_oscn'
    args.pretrained_path = os.path.join(
        './rslt', experiment_name, model_name, args.run_id)
    main(args=args)
    args = config_analyzer_vae_oscn

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
        args = copy.copy(config_trainer_vae_cmnist)
        args.seed = seed
        args.run_id = 'vae_cmnist_seed_' + str(seed)
        run_ids_dict['VAE_CMNIST'].append(args.run_id)
        if execute_train:
            main(args=args)

        args = copy.copy(config_trainer_vae_oscn)
        args.seed = seed
        args.run_id = 'vae_oscn_seed_' + str(seed)
        run_ids_dict['VAE_OSCN'].append(args.run_id)
        if execute_train:
            main(args=args)

        # MMVAE
        args = copy.copy(config_trainer_mmvae_cmnist_oscn)
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
                args = copy.copy(config_analyzer_vae_oscn)
            elif model_name == 'VAE_CMNIST':
                args = copy.copy(config_analyzer_vae_cmnist)
            elif model_name == 'MMVAE_CMNIST_OSCN':
                args = copy.copy(config_analyzer_mmvae_cmnist_oscn)
            else:
                Exception
            args.run_id = run_id
            args.pretrained_path = os.path.join(
                './rslt', experiment_name, model_name, run_id)
            main(args=args)

def test_synthesize():
    args = copy.copy(config_synthesizer)
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

    args = copy.copy(config_synthesizer)
    args.run_ids_dict = run_ids_dict
    main(args=args)

def test_run_all():
    run_all()

def test():
    # test_train_classifier()
    test_train()
    test_analyse()
    test_synthesize()
    # test_pipeline()
    # test_run_all()

if __name__ == '__main__':
    test()
