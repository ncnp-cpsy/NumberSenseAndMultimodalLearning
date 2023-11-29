# clevr model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset
import numpy as np

from src.utils import Constants
from src.vis import plot_embeddings, plot_kls_df
from src.models.vae import VAE

# Constants
dataSize = torch.Size([1, 100, 160])
imgChans = dataSize[0]
fBase = 32  # base size of filter channels


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for clevr image data. """

    def __init__(self, latent_dim):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size: 1 x 160 x 100
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 80 x 50
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 40 x 25
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 20 x 12
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 8) x 10 x 6
            nn.Conv2d(fBase * 8, fBase * 16, 2, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 16) x 6 x 4
        )
        self.c1 = nn.Conv2d(fBase * 16, latent_dim, (6,4), 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 16, latent_dim, (6,4), 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Dec(nn.Module):
    """ Generate a clevr image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 16, (6,4), 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 16) x 6 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, 4, 2, 2, bias=True),
            nn.ReLU(True),
            # size: (fBase * 16) x 10 x 6
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 20 x 12
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 40 x 25
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, (1,0), bias=True),
            nn.ReLU(True),
            # size: (fBase) x 80 x 50
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 160 x 100
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale

class clevr_dataset(Dataset):
  def __init__(self, train):
        if train:
            self.images = torch.load('../data/clevr_train_images.pt')
            self.labels = torch.load('../data/clevr_train_labels.pt')
        else:
            self.images = torch.load('../data/clevr_test_images.pt')
            self.labels = torch.load('../data/clevr_test_labels.pt')

  def __getitem__(self, index):
      return self.images[index], self.labels[index]

  def __len__(self):
      return len(self.images)

clevr_train_dataset = clevr_dataset(train = True)
clevr_test_dataset  = clevr_dataset(train = False)


class CLEVR(VAE):
    """ Derive a specific sub-class of a VAE for clevr """

    def __init__(self, params):
        super(CLEVR, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            Enc(params.latent_dim),
            Dec(params.latent_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'clevr'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = torch.utils.data.DataLoader(clevr_train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
        test = torch.utils.data.DataLoader(clevr_test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)

        """ train = DataLoader(datasets.clevr('../data', split='train', download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.clevr('../data', split='test', download=True, transform=tx),
                          batch_size=batch_size, shuffle=shuffle, **kwargs) """
        return train, test

    def generate(self, runPath, epoch):
        N, K = 4, 4
        samples = super(CLEVR, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def generate_special(self, mean):
        N = 64
        samples_list = super(CLEVR, self).generate_special(N, mean)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       './gen_special_samples_{}.png'.format(i),
                       nrow=int(sqrt(N)))

    def latent(self, data):
        zss= super(CLEVR, self).get_latent(data)
        return zss

    def reconstruct(self, data, runPath, epoch):
        recon = super(CLEVR, self).reconstruct(data[:24])
        comp = torch.cat([data[:24], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CLEVR, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
