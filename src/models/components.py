import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Constants

hidden_dim = 400

def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class EncMLP(nn.Module):
    """ Generate latent parameters for MNIST image data.
    Using multi layer perceptron.
    """

    def __init__(self,
                 latent_dim,
                 num_hidden_layers=1,
                 data_size=torch.Size([3, 32, 32]),
                 ):
        super().__init__()
        print("Encoder based on MLP was constructed.")

        data_dim = int(np.prod(data_size))
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True)))
        modules.extend([
            extra_hidden_layer() for _ in range(num_hidden_layers - 1)])

        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1) )  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta

class DecMLP(nn.Module):
    """ Generate an MNIST image given a sample from the latent space.
    Using multi layer perceptron.
    """

    def __init__(self,
                 latent_dim,
                 num_hidden_layers=1,
                 data_size=torch.Size([3, 32, 32]),
                 ):
        super().__init__()
        print("Decoder based on MLP was constructed.")
        data_dim = int(np.prod(data_size))
        self.data_size = data_size

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True)))
        modules.extend([
            extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *self.data_size))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)

        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class EncMLPSimple(nn.Module):
    """ Generate latent parameters for MNIST image data.
    Using multi layer perceptron.
    """

    def __init__(self,
                 latent_dim,
                 num_hidden_layers=None,
                 data_size=torch.Size([3, 32, 32]),
                 ):
        super().__init__()
        print("Encoder based on Simple MLP was constructed.")
        self.data_size = data_size

        data_dim = int(np.prod(data_size))
        layer_dims = [data_dim, 2048, 1024, 512, 256, 128]
        layer_dims_zip = zip(
            layer_dims[:(len(layer_dims) - 1)],
            layer_dims[1:(len(layer_dims))],
        )

        modules = []
        modules.extend([
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(True),
            ) for dim_in, dim_out in layer_dims_zip
        ])

        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(layer_dims[-1], latent_dim)
        self.fc22 = nn.Linear(layer_dims[-1], latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1) )  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), \
            F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta

class DecMLPSimple(nn.Module):
    """ Generate an MNIST image given a sample from the latent space.
    Using multi layer perceptron.
    """

    def __init__(self,
                 latent_dim,
                 num_hidden_layers=None,
                 data_size=torch.Size([3, 32, 32]),
                 ):
        super().__init__()
        print("Decoder based on Simple MLP was constructed.")
        self.data_size = data_size

        data_dim = int(np.prod(data_size))
        layer_dims = [data_dim, 2048, 1024, 512, 256, 128, latent_dim]
        layer_dims.reverse()
        layer_dims_zip = zip(
            layer_dims[:(len(layer_dims) - 1)],
            layer_dims[1:(len(layer_dims))],
        )

        modules = []
        modules.extend([
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(True),
            ) for dim_in, dim_out in layer_dims_zip
        ])

        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(layer_dims[-1], data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *self.data_size))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)

        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class EncCNN_OSCN(nn.Module):
    """ Generate latent parameters for OSCN image data. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Encoder based on CNN was constructed.")
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
        return self.c1(e).squeeze(), \
            F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class DecCNN_OSCN(nn.Module):
    """ Generate a OSCN image given a sample from the latent space. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Decoder based on CNN was constructed.")
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


class EncCNN_CMNIST(nn.Module):
    """ Generate latent parameters for CMNIST image data. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Encoder based on CNN was constructed.")
        self.enc = nn.Sequential(
            # input size: 3 x 28 x 28
            nn.Conv2d(img_chans, f_base, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base) x 14 x 14
            nn.Conv2d(f_base, f_base * 2, 4, 2, 2, bias=True),
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
        return self.c1(e).squeeze(), \
            F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class DecCNN_CMNIST(nn.Module):
    """ Generate a CMNIST image given a sample from the latent space. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Decoder based on CNN was constructed.")
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f_base * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (f_base * 4) x 4 x 4
            nn.ConvTranspose2d(f_base * 4, f_base * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (f_base * 2) x 8 x 8
            nn.ConvTranspose2d(f_base * 2, f_base, 4, 2, 2, bias=True),
            nn.ReLU(True),
            # size: (f_base) x 14 x 14
            nn.ConvTranspose2d(f_base, img_chans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 28 x 28
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale


class EncCNNAdd_OSCN(nn.Module):
    """ Generate latent parameters for OSCN image data. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Encoder based on complex CNN was constructed.")
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(img_chans, f_base, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base),
            nn.ReLU(True),

            # size: (f_base) x 16 x 16
            nn.Conv2d(f_base, f_base * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base * 2),
            nn.ReLU(True),

            # size: (f_base * 2) x 8 x 8
            nn.Conv2d(f_base * 2, f_base * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base * 4),
            nn.ReLU(True),
            # size: (f_base * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), \
            F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class DecCNNAdd_OSCN(nn.Module):
    """ Generate a OSCN image given a sample from the latent space. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Decoder based on complex CNN was constructed.")
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f_base * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(f_base * 4),
            nn.ReLU(True),

            # size: (f_base * 4) x 4 x 4
            nn.ConvTranspose2d(f_base * 4, f_base * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base * 2),
            nn.ReLU(True),

            # size: (f_base * 2) x 8 x 8
            nn.ConvTranspose2d(f_base * 2, f_base, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base),
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


class EncCNNAdd_CMNIST(nn.Module):
    """ Generate latent parameters for CMNIST image data. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Encoder based on complex CNN was constructed.")
        self.enc = nn.Sequential(
            # input size: 3 x 28 x 28
            nn.Conv2d(img_chans, f_base, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base),
            nn.ReLU(True),

            # size: (f_base) x 14 x 14
            nn.Conv2d(f_base, f_base * 2, 4, 2, 2, bias=True),
            nn.BatchNorm2d(f_base * 2),
            nn.ReLU(True),

            # size: (f_base * 2) x 8 x 8
            nn.Conv2d(f_base * 2, f_base * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base * 4),
            nn.ReLU(True),
            # size: (f_base * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(f_base * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), \
            F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class DecCNNAdd_CMNIST(nn.Module):
    """ Generate a CMNIST image given a sample from the latent space. """

    def __init__(self,
                 latent_dim,
                 img_chans=3,
                 f_base=32,
                 ):
        super().__init__()
        print("Decoder based on complex CNN was constructed.")
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f_base * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(f_base * 4),
            nn.ReLU(True),

            # size: (f_base * 4) x 4 x 4
            nn.ConvTranspose2d(f_base * 4, f_base * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(f_base * 2),
            nn.ReLU(True),

            # size: (f_base * 2) x 8 x 8
            nn.ConvTranspose2d(f_base * 2, f_base, 4, 2, 2, bias=True),
            nn.BatchNorm2d(f_base),
            nn.ReLU(True),

            # size: (f_base) x 14 x 14
            nn.ConvTranspose2d(f_base, img_chans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 28 x 28
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale
