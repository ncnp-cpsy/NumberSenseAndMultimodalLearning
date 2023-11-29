import torch
from torch import nn

from src.datasets import DatasetCMNIST, DatasetOSCN
from src.models.classifier import Classifier

class Classifier_OSCN(Classifier):
    """Classifier for OSCN
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params

        self.data_size = torch.Size([3, 32, 32])
        self.img_ch = self.data_size[0]
        self.f_base = 32  # base size of filter channels
        self.latent_dim = 8

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(self.img_ch, self.f_base, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(self.f_base, self.f_base * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (self.f_base * 2) x 8 x 8
            nn.Conv2d(self.f_base * 2, self.f_base * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.conv1 = nn.Conv2d(self.f_base * 4, self.latent_dim, 4, 1, 0, bias=True)
        # c1 size: latent_dim x 1 x 1

        self.fc1 = nn.Linear(self.latent_dim, 9)

    def forward(self, x):
        x = self.enc(x)
        x = self.conv1(x).squeeze()
        x = self.fc1(x)
        return x

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        oscn_train_dataset = DatasetOSCN(
            train=True,
            model_name='Classifier_OSCN',
            device=device,
            convert_label=True,
        )
        oscn_test_dataset = DatasetOSCN(
            train=False,
            model_name='Classifier_OSCN',
            device=device,
            convert_label=True,
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
