import torch
from torch import nn

from src.datasets import DatasetCMNIST, DatasetOSCN
from src.models.classifier import Classifier
from src.models.components import (
    EncMLP,
    EncMLPSimple,
    EncCNN_OSCN,
    EncCNNAdd_OSCN,
)

class Classifier_OSCN(Classifier):
    """Classifier for OSCN
    """
    def __init__(self, params):
        super().__init__(params)
        if 'use_cnn' in dir(params):
            use_cnn = params.use_cnn
        else:
            use_cnn = True

        data_size = torch.Size([3, 32, 32])
        img_chans = data_size[0]
        f_base = 32  # base size of filter channels

        # align with VAE
        latent_dim = params.latent_dim
        num_hidden_layers = params.num_hidden_layers

        # original model
        # use_cnn = True
        # latent_dim = 20
        # num_hidden_layers = 1

        if use_cnn is True or use_cnn == 'cnn':
            enc = EncCNN_OSCN(
                latent_dim=latent_dim,
                img_chans=img_chans,
                f_base=f_base,
            )
        elif use_cnn == 'cnn-add':
            enc = EncCNNAdd_OSCN(
                latent_dim=latent_dim,
                img_chans=img_chans,
                f_base=f_base,
            )
        elif use_cnn is False or use_cnn == 'mlp':
            enc = EncMLP(
                latent_dim=latent_dim,
                num_hidden_layers=num_hidden_layers,
                data_size=data_size
            )
        elif use_cnn == 'mlp-simple':
            enc = EncMLPSimple(
                latent_dim=params.latent_dim,
                num_hidden_layers=params.num_hidden_layers,
                data_size=data_size
            )
        else:
            Exception

        self.data_size = data_size
        self.enc = enc
        self.fc = nn.Linear(latent_dim, 9)

    def forward(self, x):
        x, _ = self.enc(x)
        x = self.fc(x)
        return x

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        oscn_train_dataset = DatasetOSCN(
            train=True,
        )
        oscn_test_dataset = DatasetOSCN(
            train=False,
        )
        train = torch.utils.data.DataLoader(
            oscn_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2)
        test = torch.utils.data.DataLoader(
            oscn_test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2)
        return train, test
