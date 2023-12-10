"""CMNIST model specification
"""
from numpy import prod, sqrt
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.utils import Constants
from src.vis import plot_embeddings, plot_kls_df
from src.datasets import DatasetCMNIST
from src.models.vae import VAE
from src.models.components import EncMLP, DecMLP, EncCNN_CMNIST, DecCNN_CMNIST


class VAE_CMNIST(VAE):
    """ Derive a specific sub-class of a VAE for CMNIST. """
    def __init__(self, params):
        if 'use_cnn' in dir(params):
            use_cnn = params.use_cnn
        else:
            use_cnn = False
        data_size = torch.Size([3, 28, 28])
        img_chans = data_size[0]
        f_base = 32

        if use_cnn:
            enc = EncCNN_CMNIST(
                latent_dim=params.latent_dim,
                img_chans=img_chans,
                f_base=f_base,
            )
            dec = DecCNN_CMNIST(
                latent_dim=params.latent_dim,
                img_chans=img_chans,
                f_base=f_base,
            )
        else:
            # In noda-san implementation, use MLP.
            enc = EncMLP(
                latent_dim=params.latent_dim,
                num_hidden_layers=params.num_hidden_layers,
                data_size=data_size
            )
            dec = DecMLP(
                latent_dim=params.latent_dim,
                num_hidden_layers=params.num_hidden_layers,
                data_size=data_size,
            )
        super().__init__(
            prior_dist=dist.Normal,
            likelihood_dist=dist.Normal,
            post_dist=dist.Normal,
            enc=enc,
            dec=dec,
            params=params
        )
        grad = {'requires_grad': params.learn_prior}

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'cmnist'
        self.data_size = data_size
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], \
            F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        """
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        train = DataLoader(
            datasets.CMNIST('../data', train=True, download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        test = DataLoader(
            datasets.CMNIST('../data', train=False, download=True, transform=tx),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)
        """
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

    def generate(self,
                 num_data=64,
                 K=9,
                 output_dir=None,
                 suffix='',
                 ):
        samples = super().generate(
            num_data=num_data,
            K=K).cpu()
        # wrangle things so they come out tiled
        # num_data x K x 1 x 28 x 28
        samples = samples.view(K, num_data, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        if output_dir is not None:
            fname = '{}/gen_samples_{:03d}.png'.format(output_dir, suffix)
            save_image(torch.stack(s), fname, nrow=int(sqrt(num_data)))
        return samples

    def generate_special(self,
                         mean,
                         num_data=64,
                         output_dir=None,
                         suffix='',
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
                fname = '{}/gen_special_samples_cmnist_{}'.format(
                    output_dir, i) + "_" + suffix + '.png'
                save_image(samples, fname, nrow=int(sqrt(num_data)))
        return samples

    def latent(self, data):
        zss= super().get_latent(data)
        return zss

    def reconstruct(self,
                    data,
                    output_dir=None,
                    suffix='',
                    num_data=None
                    ):
        if num_data is not None:
            data = data[:num_data]
        recon = super().reconstruct(data)
        comp = torch.cat([data, recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(output_dir, suffix))
        return recon

    def analyse(self,
                data,
                output_dir=None,
                suffix='',
                ):
        zemb, zsl, kls_df = super().analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        if output_dir is not None:
            fname = '{}/emb_umap_{:03d}.png'.format(output_dir, suffix)
            plot_embeddings(zemb, zsl, labels, fname)
            fname = '{}/kl_distance_{:03d}.png'.format(output_dir, suffix)
            plot_kls_df(kls_df, fname)
        return zemb, zsl, kls_df
