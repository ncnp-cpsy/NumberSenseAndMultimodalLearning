# objectives of choice
import numpy as np
from numpy import prod

import torch
import torch.nn.functional as F

from src.utils import log_mean_exp, is_multidata, kl_divergence
from src.datasets import convert_label_to_int

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def elbo(model, x, K=1, conditional=False, labels=None, device=None):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()


def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x


def iwae(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None, conditional = False, labels = None, device = None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


# multi-modal variants
def m_elbo_naive(model, x, K=1):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def m_elbo(model, x, K=1,  conditional = False, labels = None, device = None):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    #v0 = 1.0 / qz_xs[0].variance.mean().item()
    #v1 = 1.0 / qz_xs[1].variance.mean().item()
    #ratio = [0.7 , 0.3]
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params)) #* ratio[r]
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1) #* ratio[r]
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    
    obj = (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0)) * (1 / len(model.vaes)) 
    return obj.mean(0).sum() #, ratio


def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, K=1, conditional = False):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    if conditional:
        qz_xs, px_zs, zss, conds_number, conds_color = model(x, K)
    else:
        qz_xs, px_zs, zss = model(x, K)

    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    #torch.cat(cond_probs) は (256,10) -> (128, 10)が2つ繋がってる

    if conditional:
        return torch.cat(lws), torch.cat(zss), torch.cat(conds_number), torch.cat(conds_color)
    else:
        return torch.cat(lws), torch.cat(zss)
    


def m_dreg(model, x, K=1, conditional = False, labels = None, device = None):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])

    if conditional:
        lw, zss, conds_numbers, conds_colors = zip(*[_m_dreg(model, _x, K, conditional = True) for _x in x_split]) #cond_probsは長さ1のリスト
        cond_number = conds_numbers[0]
        cond_color = conds_colors[0]
        labels = torch.Tensor(np.array(list(labels))).to(dtype=torch.long).to(device)
    else:
        lw, zss = zip(*[_m_dreg(model, _x, K) for _x in x_split])
    
    

    lw = torch.cat(lw, 1)  # concat on batch 結果, torch.Size([40, 128])に
    zss = torch.cat(zss, 1)  # concat on batch　結果、torch.Size([40, 128])に。
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp() # torch.Size([40, 128])
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    if conditional:
        num_classify_loss = F.nll_loss(cond_number, labels)
        col_classify_loss = F.nll_loss(cond_color, labels)
    
    #TODO : ここに、識別のタームを足す。cond_probs[0]が(256,10)のテンソルだから、それとlabelsを使って識別ロスを普通に求めれればOK。
    #print((grad_wt * lw).sum() , classify_loss )
    if conditional :
        return (grad_wt * lw).sum() - num_classify_loss * 10000 - col_classify_loss * 10000
    else:
        return (grad_wt * lw).sum() 

def _m_dreg_looser(model, x, K=1, conditional = False):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    if conditional:
        qz_xs, px_zs, zss, conds_numbers, conds_colors = model(x, K)
    else:
        qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)

    if conditional:
        return torch.stack(lws), torch.stack(zss), conds_numbers, conds_colors 
    else:
        return torch.stack(lws), torch.stack(zss)


def m_dreg_looser(model, x, K=1, conditional = False, labels = None, device = None):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])

    if conditional:
        lw, zss, conds_numbers, conds_colors= zip(*[_m_dreg_looser(model, _x, K, conditional = conditional) for _x in x_split]) #cond_probsは長さ1のリスト
        conds_number = conds_numbers[0]
        conds_color = conds_colors[0]
    else:
        lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    
    if labels != None:
        labels_number = torch.Tensor(np.array(labels[0])).to(dtype=torch.long).to(device)
        labels_color = torch.Tensor(np.array(labels[1])).to(dtype=torch.long).to(device)

    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    
    if conditional : 
        #print(conds.shape, labels.shape)
        classify_loss_number = F.nll_loss(conds_number, labels_number)
        classify_loss_color = F.nll_loss(conds_color, labels_color)

        return (grad_wt * lw).mean(0).sum() - classify_loss_number * 10000 - classify_loss_color * 10000
    else:
        return (grad_wt * lw).mean(0).sum()


def cross(model, x, K=1, conditional=False, labels=None, device=None):
    """Cross Entropy Loss for Classifier

    Note
    ----
    Pattern A
    > x = torch.randn(3, 5)
    > labels = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).to(torch.float)
    > loss = F.cross_entropy(x, labels)

    Pattern B
    > labels = torch.argmax(labels, dim=1)
    > loss = F.nll_loss(F.log_softmax(x, dim=1), labels)
    """
    pred = model(x)
    labels = convert_label_to_int(
        label=labels,
        model_name=model.__class__.__name__,
        target_property=1,
    )
    labels = torch.tensor(labels).to(device) - 1
    # num_class = 9
    # label = [l - 1 for l in label]
    # label = np.identity(num_class)[label]
    accuracy = sum(torch.argmax(pred, dim=1) == labels).item()
    print(accuracy)
    # print(
    #     '\ninput:', x.shape,
    #     '\nlabels in objectives:', labels,
    #     '\npred:', pred,
    # )
    loss = F.nll_loss(F.log_softmax(pred, dim=1), labels)
    return - loss
