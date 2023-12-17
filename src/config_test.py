"""Parameters for model and training
"""
from src.utils import DotDict

epochs = 3
experiment_name = 'test_loop_1'
id_vae_cmnist = 'test_vae_cmnist'
id_vae_oscn = 'test_vae_oscn'
id_mmvae_cmnist_oscn = 'test_mmvae_cmnist_oscn'
id_classifier_cmnist = 'test_classifier_cmnist'
id_classifier_oscn = 'test_classifier_oscn'

config_test_trainer_vae_cmnist = DotDict({
    'experiment': experiment_name,
    'model': 'VAE_CMNIST',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 2,
    'use_conditional': False,
    'use_cnn': 'cnn-add',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'elbo',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_test_trainer_vae_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'VAE_OSCN',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 2,
    'use_conditional': False,
    'use_cnn': 'cnn-add',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'elbo',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_test_trainer_mmvae_cmnist_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'MMVAE_CMNIST_OSCN',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 2,
    'use_conditional': False,
    'use_cnn': 'cnn-add',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'dreg',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': False,  # true in noda-san experiment
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_test_analyzer_vae_cmnist = DotDict({
    'run_type': 'analyse',
    'run_id': id_vae_cmnist,
    'pretrained_path': './rslt/' + experiment_name + '/VAE_CMNIST/' + id_vae_cmnist,
})

config_test_analyzer_vae_oscn = DotDict({
    'run_type': 'analyse',
    'run_id': id_vae_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/VAE_OSCN/' + id_vae_oscn,
})

config_test_analyzer_mmvae_cmnist_oscn = DotDict({
    'run_type': 'analyse',
    'run_id': id_mmvae_cmnist_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/MMVAE_CMNIST_OSCN/' + id_mmvae_cmnist_oscn,
})

config_test_classifier_cmnist = DotDict({
    'experiment': experiment_name,
    'model': 'Classifier_CMNIST',
    'run_type': 'train',
    'run_id': id_classifier_cmnist,
    'pretrained_path': './rslt/' + experiment_name + '/Classifier_CMNIST/' + id_classifier_cmnist,
    'seed': 4,
    # Architecture
    'num_hidden_layers': 2,
    'latent_dim': 20,
    'use_conditional': False,
    'use_cnn': 'cnn-add',
    # Training and Loss
    'obj': 'cross',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})


config_test_classifier_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'Classifier_OSCN',
    'run_type': 'train',
    'run_id': id_classifier_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/Classifier_OSCN/' + id_classifier_oscn,
    'seed': 4,
    # Architecture
    'num_hidden_layers': 2,
    'latent_dim': 20,
    'use_conditional': False,
    'use_cnn': 'cnn-add',
    # Training and Loss
    'obj': 'cross',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
    'device': 'cuda',
})

config_test_synthesizer = DotDict({
    'experiment': experiment_name,
    'run_type': 'synthesize',
    'pretrained_path': './rslt/' + experiment_name + '/VAE_OSCN/' + id_vae_oscn,
    'run_id': id_vae_oscn,
    'no_cuda': False,
    'device': 'cuda',
})


