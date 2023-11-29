# CMNIST-OSCN multi-modal model specification
import os

from numpy import sqrt, prod
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

from src.vis import plot_embeddings, plot_kls_df
from src.models.mmvae import MMVAE
from src.models.vae_cmnist import VAE_CMNIST
from src.models.vae_oscn import VAE_OSCN


class MMVAE_CMNIST_OSCN(MMVAE):
    def __init__(self, params):
        super(MMVAE_CMNIST_OSCN, self).__init__(
            dist.Laplace, params, CMNIST, OSCN)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'cmnist-oscn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # get transformed indices
        t_cmnist = torch.load('../data/train-ms-cmnist-idx.pt')
        t_oscn = torch.load('../data/train-ms-oscn-idx.pt')
        s_cmnist = torch.load('../data/test-ms-cmnist-idx.pt')
        s_oscn = torch.load('../data/test-ms-oscn-idx.pt') 

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        train_cmnist_oscn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_cmnist[i], size=len(t_cmnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_oscn[i], size=len(t_oscn))
        ])
        test_cmnist_oscn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_cmnist[i], size=len(s_cmnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_oscn[i], size=len(s_oscn))
        ]) 

        
        """ train_cmnist_oscn = TensorDataset([t1.dataset, t2.dataset])
        test_cmnist_oscn = TensorDataset([s1.dataset, s2.dataset]) """
        print(len(train_cmnist_oscn), len(test_cmnist_oscn) )

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_cmnist_oscn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_cmnist_oscn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(MMVAE_CMNIST_OSCN, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def generate_special(self, mean, target_modality, label):
        N = 64
        samples_list = super(MMVAE_CMNIST_OSCN, self).generate_special(N, mean)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            if i == target_modality:
                save_image(samples,
                        'addition_images/gen_special_samples_cmnist-oscn_{}'.format(i) + "_" + label + '.png',
                        nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch, n = 8):
        recons_mat = super(MMVAE_CMNIST_OSCN, self).reconstruct([d[:n] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:n].cpu()
                recon = recon.squeeze(0).cpu()
                # resize Cmnist to 32 and colour. 0 => Cmnist, 1 => OSCN
                _data = _data if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                recon = recon if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch):
        #zemb, zsl, kls_df = super(MMVAE_CMNIST_OSCN, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        print(labels)
        #plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        #plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

    def latent(self, data):
        zss= super(MMVAE_CMNIST_OSCN, self).get_latent(data)
        return zss

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
