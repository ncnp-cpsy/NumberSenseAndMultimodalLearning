from numpy import prod
import torch
from torch import nn

from src.datasets import DatasetCMNIST, DatasetOSCN
from src.models.classifier import Classifier

class Classifier_CMNIST(Classifier):
    """Classifier for CMNIST
    """
    def __init__(self, params):
        super().__init__(params)

        self.params = params
        hidden_dim = 400
        latent_dim = 8
        num_hidden_layers=1

        self.data_size = torch.Size([3, 28, 28])
        self.img_ch = self.data_size[0]
        self.f_base = 32  # base size of filter channels

        def extra_hidden_layer():
            return nn.Sequential(nn.Linear(
                hidden_dim, hidden_dim), nn.ReLU(True))

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(int(prod(self.data_size)), hidden_dim),
            nn.ReLU(True)))
        modules.extend([
            extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 9)

    def forward(self, x):
        x = self.enc(x.view(*x.size()[:-3], -1) )  # flatten data
        x = self.fc1(x)
        x = self.fc2(x)
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
