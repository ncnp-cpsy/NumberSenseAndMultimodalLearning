"""Base class of classifier for the number detection
"""
import torch
from torch import nn


class Classifier(nn.module):
    """Base classifier class
    """

    def __init__(self, params):
        super().__init__()

    def forward(self):
        raise NotImplementedError()

    def reconstruct(self, data, runPath, epoch):
        """Predictions of class labels
        """
        self.eval()
        with torch.no_grad():
            pred = self(data)
        return pred

    def generate(self, runPath, epoch):
        """Predictions of class labels
        """
        pass

    def analyze(self, data, runPath, epoch):
        pass

    @staticmethod
    def getDataLoaders():
        raise NotImplementedError()
