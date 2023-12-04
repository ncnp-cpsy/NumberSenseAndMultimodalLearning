"""OSCN model specification
"""

import numpy as np
from numpy import sqrt
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from src.datasets import DatasetOSCN
from src.utils import Constants
from src.vis import plot_embeddings, plot_kls_df
from src.models.vae import VAE


# Constants
data_size = torch.Size([3, 32, 32])
img_chans = data_size[0]
f_base = 32  # base size of filter channels


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for OSCN image data. """

    def __init__(self, latent_dim):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(img_chans, f_base, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base) x 16 x 16
            nn.Conv2d(f_base, f_base * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base * 2) x 8 x 8
            nn.Conv2d(f_base * 2, f_base * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Dec(nn.Module):
    """ Generate a OSCN image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f_base * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (f_base * 4) x 4 x 4
            nn.ConvTranspose2d(f_base * 4, f_base * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base * 2) x 8 x 8
            nn.ConvTranspose2d(f_base * 2, f_base, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base) x 16 x 16
            nn.ConvTranspose2d(f_base, img_chans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale



class VAE_OSCN(VAE):
    """ Derive a specific sub-class of a VAE for OSCN """

    def __init__(self, params):
        super(VAE_OSCN, self).__init__(
            prior_dist=dist.Laplace,
            # likelihood_dist=dist.Laplace,
            likelihood_dist=dist.Normal,
            post_dist=dist.Laplace,
            enc=Enc(params.latent_dim),
            dec=Dec(params.latent_dim),
            params=params,
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'oscn'
        self.data_size = data_size
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return (
            self._pz_params[0],
            F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
        )

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        oscn_train_dataset = DatasetOSCN(train=True)
        oscn_test_dataset = DatasetOSCN(train=False)
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
        """
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = DataLoader(
            datasets.OSCN('../data', split='train', download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)
        test = DataLoader(
            datasets.OSCN('../data', split='test', download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)
        """
        return train, test

    def generate(self,
                 num_data=64,
                 K=9,
                 output_dir=None,
                 suffix='',
                 ):
        samples = super().generate(num_data, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, num_data, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        if output_dir is not None:
            fname = '{}/gen_samples_{:03d}.png'.format(output_dir, suffix)
            save_image(torch.stack(s), fname, nrow=int(sqrt(num_data)))
        return samples

    def generate_special(self,
                         mean,
                         label,
                         output_dir=None,
                         num_data=64
                         ):
        samples_list = super().generate_special(
            mean=mean,
            num_data=num_data,
        )
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(num_data, *samples.size()[1:])
            if output_dir is not None:
                fname = '{}/gen_special_samples_oscn_'.format(output_dir) + label + '.png'
                save_image(samples, fname, nrow=int(sqrt(num_data)))
        return samples

    def reconstruct(self,
                    data,
                    num_data=None,
                    output_dir=None,
                    suffix='',
                    ):
        if num_data is not None:
            data = data[:num_data]
        recon = super().reconstruct(data=data)
        if output_dir is not None:
            composed = torch.cat([data, recon]).data.cpu()
            fname = '{}/recon_{:03d}.png'.format(output_dir, suffix)
            save_image(composed, fname)
        return recon

    def latent(self, data):
        zss= super().get_latent(data)
        return zss

    def analyse(self,
                data,
                output_dir=None,
                suffix='',
                ):
        zemb, zsl, kls_df = super().analyse(data, K=10)
        if output_dir is not None:
            labels = ['Prior', self.modelName.lower()]
            fname = '{}/emb_umap_{:03d}.png'.format(output_dir, suffix)
            plot_embeddings(zemb, zsl, labels, fname)
            fname = '{}/kl_distance_{:03d}.png'.format(output_dir, suffix)
            plot_kls_df(kls_df, fname)
        return zemb, zsl, kls_df
