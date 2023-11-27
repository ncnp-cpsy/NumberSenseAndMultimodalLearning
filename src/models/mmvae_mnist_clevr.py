# MNIST-clevr multi-modal model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE
from .vae_mnist import MNIST
from .vae_clevr import CLEVR


class MNIST_CLEVR(MMVAE):
    def __init__(self, params):
        super(MNIST_CLEVR, self).__init__(dist.Laplace, params, MNIST, CLEVR)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'mnist-clevr'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        
        # get transformed indices
        t_mnist = torch.load('../data/train-ms-mnist-idx.pt')
        t_clevr = torch.load('../data/train-ms-clevr-idx.pt')
        s_mnist = torch.load('../data/test-ms-mnist-idx.pt')
        s_clevr = torch.load('../data/test-ms-clevr-idx.pt') 

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        train_mnist_clevr = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_clevr[i], size=len(t_clevr))
        ])
        test_mnist_clevr = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_clevr[i], size=len(s_clevr))
        ]) 

        
        """ train_mnist_clevr = TensorDataset([t1.dataset, t2.dataset])
        test_mnist_clevr = TensorDataset([s1.dataset, s2.dataset]) """
        print(len(train_mnist_clevr), len(test_mnist_clevr) )

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_clevr, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_clevr, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(MNIST_CLEVR, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def generate_special(self, mean):
        N = 64
        samples_list = super(MNIST_CLEVR, self).generate_special(N, mean)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       './gen_special_samples_{}.png'.format(i),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch, n = 8):
        recons_mat = super(MNIST_CLEVR, self).reconstruct([d[:n] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:n].cpu()
                recon = recon.squeeze(0).cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => clevr

                #ここ買えたよ
                #_data = _data if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                #recon = recon if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                #comp = torch.cat([_data, recon])
                save_image(_data, '{}/recon_{}x{}_{:03d}_moto.png'.format(runPath, r, o, epoch))
                save_image(recon, '{}/recon_{}x{}_{:03d}_saki.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch):
        #zemb, zsl, kls_df = super(MNIST_CLEVR, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        print(labels)
        #plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        #plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

    def latent(self, data):
        zss= super(MNIST_CLEVR, self).get_latent(data)
        return zss

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
