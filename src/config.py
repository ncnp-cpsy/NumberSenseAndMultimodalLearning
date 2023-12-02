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
    'epochs': 30,
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
    'pre_trained': './rslt/test/Classifier_CMNIST/2023-12-01T23:18:21.255314zhkqwh4b',
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
    'epochs': 30,
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
    # 'pre_trained': '',
    'pre_trained': './rslt/test/Classifier_OSCN/2023-12-01T23:16:41.595779y56i2qp2',
    'print_freq': 0,
    'run_type': 'train',
    'seed': 4,
    'use_conditional': False,
})
