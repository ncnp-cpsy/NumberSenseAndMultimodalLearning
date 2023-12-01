"""Base class of classifier for the number detection
"""
import torch
from torch import nn


class Classifier(nn.Module):
    """Base classifier class
    """

    def __init__(self, params):
        super().__init__()

    @staticmethod
    def getDataLoaders():
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def generate(self, run_path, epoch):
        """Predictions of class labels
        """
        pass

    def reconstruct(self, data, run_path, epoch):
        """Predictions of class labels
        """
        self.eval()
        with torch.no_grad():
            pred = self(data)
        return pred

    def analyse(self, data, run_path, epoch):
        pass
