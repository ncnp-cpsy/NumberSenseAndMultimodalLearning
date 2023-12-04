# MNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.utils import Constants
from src.vis import plot_embeddings, plot_kls_df
from src.models.vae import VAE

# Constants
data_size = torch.Size([1, 28, 28])
data_dim = int(prod(data_size))
hidden_dim = 400


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Dec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *data_size))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)

        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class VAE_MNIST(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """

    def __init__(self, params):
        super().__init__(
            dist.Normal, # prior
            dist.Normal,  # likelihood
            dist.Normal,  # posterior
            Enc(params.latent_dim, params.num_hidden_layers),
            Dec(params.latent_dim, params.num_hidden_layers),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'mnist'
        self.data_size = data_size
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], \
            F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        train = DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        test = DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        return train, test

    def generate(self,
                 num_data=64,
                 K=9,
                 run_path=None,
                 suffix='',
                 ):
        samples = super().generate(num_data, K).cpu()
        # wrangle things so they come out tiled
        # N x K x 1 x 28 x 28
        samples = samples.view(K, num_data, *samples.size()[1:]).transpose(0, 1)
        if run_path is not None:
            s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
            fname = '{}/gen_samples_{:03d}.png'.format(run_path, suffix)
            save_image(torch.stack(s), fname, nrow=int(sqrt(num_data)))
        return samples

    def latent(self, data):
        zss= super().get_latent(data)
        return zss

    def reconstruct(self,
                    data,
                    run_path=None,
                    suffix='',
                    num_data=None):
        if num_data is not None:
            data = data[:num_data]
        recon = super().reconstruct(data)
        comp = torch.cat([data, recon]).data.cpu()
        if run_path is not None:
            fname = '{}/recon_{:03d}.png'.format(run_path, suffix)
            save_image(comp, fname)
        return recon

    def analyse(self, data, run_path, suffix):
        zemb, zsl, kls_df = super().analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, suffix))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(run_path, suffix))
        return zemb, zsl, kls_df
