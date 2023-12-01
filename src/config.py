class DotDict(dict):
    """Dictionary which can be accessed using dot symbol
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


config_classifier_cmnist = DotDict({
    'K': 20,
    'batch_size': 128,
    'cuda': True,
    'device': 'cuda',
    'epochs': 10,
    'experiment': './test/Classifier_CMNIST',
    'latent_dim': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    'model': 'Classifier_CMNIST',
    'no_analytics': False,
    'no_cuda': False,
    'num_hidden_layers': 1,
    'obj': 'cross',
    # 'pre_trained': '',
    # 'pretrained_path': '',
    'pre_trained': './rslt/test/Classifier_CMNIST/2023-12-01T15:43:23.118510rruaoa16/',
    'pretrained_path': './rslt/test/Classifier_CMNIST/2023-12-01T15:43:23.118510rruaoa16/',
    'print_freq': 0,
    'run_type': 'train',
    'seed': 4,
    'use_conditional': False,
})


config_classifier_oscn = DotDict({
    'K': 20,
    'batch_size': 128,
    'cuda': True,
    'device': 'cuda',
    'epochs': 10,
    'experiment': 'test/Classifier_OSCN',
    'latent_dim': 20,
    'learn_prior': False,
    'llik_scaling': 0.0,
    'logp': False,
    'looser': False,
    'model': 'Classifier_OSCN',
    'no_analytics': False,
    'no_cuda': False,
    'num_hidden_layers': 1,
    'obj': 'cross',
    # 'pre_trained': '.',
    # 'pretrained_path': '',
    'pre_trained': './rslt/test/Classifier_OSCN/2023-12-01T15:42:21.987845yxi7qjfw',
    'pretrained_path': './rslt/test/Classifier_OSCN/2023-12-01T15:42:21.987845yxi7qjfw',
    'print_freq': 0,
    'run_type': 'train',
    'seed': 4,
    'use_conditional': False,
})
