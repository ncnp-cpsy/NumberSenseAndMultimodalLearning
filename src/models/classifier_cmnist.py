import torch
from torch import nn

from src.datasets import DatasetCMNIST, DatasetOSCN
from src.models.classifier import Classifier

class ClassifierCMNIST(Classifier):
    """Classifier for CMNIST
    """
    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        cmnist_train_dataset = DatasetCMNIST(train=True)
        cmnist_test_dataset = DatasetCMNIST(train=False)
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
