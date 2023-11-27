# sMNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
dataSize = torch.Size([3, 28, 28])
imgChans = dataSize[0]
fBase = 32

data_dim = int(prod(dataSize))
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
        e = self.enc(x.view(*x.size()[:-3], -1) )  # flatten data
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
        d = torch.sigmoid(p.view(*z.size()[:-1], *dataSize))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)

        return d, torch.tensor(0.75).to(z.device)  # mean, length scale



## write code here
class smnist_dataset(Dataset):
  def __init__(self, mode):
        if mode == 'train':
            self.images = torch.load('../data/smnist_train_images.pt')
            self.labels = torch.load('../data/smnist_train_labels.pt')
        elif mode =='test':
            self.images = torch.load('../data/smnist_test_images.pt')
            self.labels = torch.load('../data/smnist_test_labels.pt')
        elif mode =='abtest':
            self.images = torch.load('../data/smnist_abtest_images.pt')
            self.labels = torch.load('../data/smnist_abtest_labels.pt')

  def __getitem__(self, index):
      return self.images[index], self.labels[index]

  def __len__(self):
      return len(self.images)

smnist_train_dataset = smnist_dataset(mode = 'train')
smnist_test_dataset  = smnist_dataset(mode = 'test')
smnist_abtest_dataset = smnist_dataset(mode = 'abtest')

class SMNIST(VAE):
    """ Derive a specific sub-class of a VAE for sMNIST. """

    def __init__(self, params):
        super(SMNIST, self).__init__(
            dist.Normal,  # prior
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
        self.modelName = 'smnist'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def save_image(array, name):
        res = array.cpu().detach().numpy().T
        tar = (res * 255).astype(np.uint8)
        if tar.shape[2] == 1:
            tar = tar[:,:,0]
        pil_image = Image.fromarray(tar) 
        pil_image.save(name)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        #train = DataLoader(datasets.sMNIST('../data', train=True, download=True, transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
        #test = DataLoader(datasets.sMNIST('../data', train=False, download=True, transform=tx),batch_size=batch_size, shuffle=shuffle, **kwargs)
        train = torch.utils.data.DataLoader(smnist_train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
        test = torch.utils.data.DataLoader(smnist_test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
        abtest = torch.utils.data.DataLoader(smnist_abtest_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)

        for i, dataT in enumerate(abtest):
            save_image(dataT[0][0], 'hayameni_check_abtest.png')
            if i == 0:
                break

        return train, test, abtest

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(SMNIST, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(SMNIST, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

    def latent(self, data):
        zss= super(SMNIST, self).get_latent(data)
        return zss

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(SMNIST, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
