from numpy import prod
import torch
from torch import nn

from src.datasets import DatasetCMNIST, DatasetOSCN
from src.models.classifier import Classifier
from src.models.components import EncMLP, EncCNN_CMNIST

class Classifier_CMNIST(Classifier):
    """Classifier for CMNIST
    """
    def __init__(self, params):
        super().__init__(params)
        use_cnn = params.use_cnn
        data_size = torch.Size([3, 28, 28])
        img_chans = data_size[0]
        f_base = 32

        if use_cnn:
            enc = EncCNN_CMNIST(
                latent_dim=params.latent_dim,
                img_chans=img_chans,
                f_base=f_base,
            )
        else:
            enc = EncMLP(
                latent_dim=params.latent_dim,
                num_hidden_layers=params.num_hidden_layers,
                data_size=data_size
            )
        self.data_size = data_size
        self.enc = enc
        self.fc = nn.Linear(params.latent_dim, 9)

    def forward(self, x):
        x, _ = self.enc(x)
        x = self.fc(x)
        return x

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        cmnist_train_dataset = DatasetCMNIST(
            train=True,
        )
        cmnist_test_dataset = DatasetCMNIST(
            train=False,
        )
        train = torch.utils.data.DataLoader(
            cmnist_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2)
        test = torch.utils.data.DataLoader(
            cmnist_test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2)

        return train, test
