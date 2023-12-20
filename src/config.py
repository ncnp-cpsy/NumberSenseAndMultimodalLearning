"""Parameters for model and training

Details were shown in the tail of this file.
Additionally, refer to the source code of MMVAE in the original paper.
"""
from src.utils import DotDict


epochs = 50
# experiment_name = 'parameter-search-17'
experiment_name = 'parameter-search-36'
id_vae_cmnist = 'vae_cmnist'
id_vae_oscn = 'vae_oscn'
id_mmvae_cmnist_oscn = 'mmvae_cmnist_oscn'
id_classifier_cmnist = 'classifier-cmnist'
id_classifier_oscn = 'classifier-oscn'

config_trainer_vae_cmnist = DotDict({
    'experiment': experiment_name,
    'model': 'VAE_CMNIST',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 3,
    'use_conditional': False,
    'use_cnn': 'mlp-simple',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'dreg',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': True,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_trainer_vae_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'VAE_OSCN',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 3,
    'use_conditional': False,
    'use_cnn': 'mlp-simple',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'dreg',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': True,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_trainer_mmvae_cmnist_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'MMVAE_CMNIST_OSCN',
    'run_type': 'train',
    'seed': 4,
    # Architecture
    'num_hidden_layers': 3,
    'use_conditional': False,
    'use_cnn': 'mlp-simple',
    'latent_dim': 20,
    # Training and Loss
    'obj': 'dreg',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': True,  # true in noda-san experiment
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})

config_analyzer_vae_cmnist = DotDict({
    'run_type': 'analyse',
    'run_id': id_vae_cmnist,
    'pretrained_path': './rslt/' + experiment_name + '/VAE_CMNIST/' + id_vae_cmnist,
})


config_analyzer_vae_oscn = DotDict({
    'run_type': 'analyse',
    'run_id': id_vae_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/VAE_OSCN/' + id_vae_oscn,
})


config_analyzer_mmvae_cmnist_oscn = DotDict({
    'run_type': 'analyse',
    'run_id': id_mmvae_cmnist_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/MMVAE_CMNIST_OSCN/' + id_mmvae_cmnist_oscn,
})


config_classifier_cmnist = DotDict({
    'experiment': experiment_name,
    'model': 'Classifier_CMNIST',
    'run_type': 'train',
    'run_id': id_classifier_cmnist,
    'pretrained_path': './rslt/' + experiment_name + '/Classifier_CMNIST/' + id_classifier_cmnist,
    'seed': 4,
    # Architecture
    'num_hidden_layers': 3,
    'latent_dim': 20,
    'use_conditional': False,
    'use_cnn': 'mlp-simple',
    # Training and Loss
    'obj': 'cross',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': True,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
})


config_classifier_oscn = DotDict({
    'experiment': experiment_name,
    'model': 'Classifier_OSCN',
    'run_type': 'train',
    'run_id': id_classifier_oscn,
    'pretrained_path': './rslt/' + experiment_name + '/Classifier_OSCN/' + id_classifier_oscn,
    'seed': 4,
    # Architecture
    'num_hidden_layers': 3,
    'latent_dim': 20,
    'use_conditional': False,
    'use_cnn': 'mlp-simple',
    # Training and Loss
    'obj': 'cross',
    'batch_size': 128,
    'epochs': epochs,
    'K': 20,
    'learn_prior': True,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    # others
    'print_freq': 100,
    'no_analytics': False,
    'no_cuda': False,
    'device': 'cuda',
})

config_synthesizer = DotDict({
    'experiment': experiment_name,
    'run_type': 'synthesize',
    'pretrained_path': './rslt/' + experiment_name + '/VAE_OSCN/' + id_vae_oscn,
    'run_id': id_vae_oscn,
    'no_cuda': False,
    'device': 'cuda',
})


""" Arguments parser and help documents in original code.
parser = argparse.ArgumentParser(description='Multi-Modal VAEs')

# Experiment
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--run-type', type=str, default='train', metavar='R',
                    choices=['train', 'classify', 'analyse', 'synthesize'],
                    help='types of run (default: train)')
parser.add_argument('--model', type=str, default='VAE_OSCN', metavar='M',
                    choices=[s.__name__ for s in models.__all__],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')

# Model settings
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')

# Loss
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
"""
