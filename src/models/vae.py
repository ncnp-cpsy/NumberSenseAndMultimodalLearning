# Base VAE class definition

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_mean, kl_divergence
from src.vis import embed_umap, tensors_to_df


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        """
        if self.params.use_conditional:
            return qz_x, px_z, zs, F.log_softmax(
                F.relu(self.sub_net(qz_x.rsample())))
        else:
        """
        return qz_x, px_z, zs

    def get_latent(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        #zs = qz_x.rsample(torch.Size([K]))
        zs = qz_x.rsample()
        #return  [zs]
        return  [self._qz_x_params[0]]

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(latents))
            data = px_z.sample(torch.Size([K]))
        return data.view(-1, *data.size()[3:])

    def generate_special(self, N, mean):
        self.eval()
        with torch.no_grad():
            data = []
            hoge = (self.pz_params)
            mean = torch.tensor(mean).reshape((1, mean.shape[0]))
            device = torch.device("cuda")
            mean = mean.to(device)
            pz = self.pz(mean, hoge[1])
            #pz = self.pz(mean, torch.zeros(1, 20).to(device) + 0.5 )
            latents = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(latents))
            data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon

    def analyse(self, data, K):
        self.eval()
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data.size(0)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
