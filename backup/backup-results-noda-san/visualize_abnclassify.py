import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import torch
from torch import optim
from mpl_toolkits.mplot3d import Axes3D
from abn_dataloader import cmnist_abn_loader , oscn_abn_loader, cmnist_oscn_abn_loader

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=5, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=0, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=2, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# args
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

if not args.experiment:
    args.experiment = model.modelName


# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))
t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')


def train(epoch, agg):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))

if __name__ == '__main__':
    if True:
        model.eval()
        with torch.no_grad():
            for i, dataT in enumerate(cmnist_oscn_abn_loader):
                data = unpack_data(dataT, device=device)
                #loss = -t_objective(model, data, K=args.K)
                if i == 0:
                    model.reconstruct(data, '.', 999, n =64)
                    model.generate('.', 999)
                    exit()
        
    if not True:
        color_dict = {
            0 : "b",
            1 : "g",
            2 : "r",
            3 : "c",
            4 : "m",
            5 : "y",
            6 : "k",
            7 : '#377eb8',
            8 : '#e41a1c',
            9 : "darkviolet"
        }

        color_dict = {
            0 : "r",
            1 : "b", #青
            2 : "g", #緑
            3 : "r", #赤
            4 : "aqua", #水色
            5 : "#a65628", #茶色
            6 : "k", #黒
            7 : 'coral', #オレンジ
            8 : 'y', #黄色
            9 : "darkviolet" #濃い紫
        } 

        """ color_dict = {
            0 : "0.1",
            1 : "0.1",
            2 : "0.2",
            3 : "0.3",
            4 : "0.4",
            5 : "0.5",
            6 : "0.6",
            7 : "0.7",
            8 : "0.8",
            9 : "0.9"
        } """


        agg = defaultdict(list)
        model.eval()
        
        label_all = []
        latent_all = []
        # cmnist か oscn か、ということ。0ならcmnist, 1ならoscn 
        target_modality = 1

        if args.model != 'cmnist_oscn' :
            target_modality = 0
        
        # その中で 0 : 色、　1 : 数、 2: 形
        n_modality = 1
        
        require_ps = False
        require_2d = False
        require_3d = False

        for i, dataT in enumerate(train_loader):
            data,label = unpack_data(dataT, device=device, require_label = True, target = target_modality)
            start_ind, end_ind = 1, 10

            #label が [0,1,3,,,]みたいに出てくる場合
            if (args.model == 'cmnist_oscn' and target_modality == 0) or (args.model == 'cmnist') :
            
                if n_modality == 0:
                    label = []
                    start_ind, end_ind = 0, 3

                    if args.model == 'cmnist_oscn' : 
                        for j in range(data[0].shape[0]):
                            dataum = data[0][j]
                            sum0, sum1, sum2 = dataum[0].sum().item(), dataum[1].sum().item(), dataum[2].sum().item()
                            if sum0 == sum1 and sum1 == sum2:
                                label.append(0)  #white 
                            elif sum1 == 0.0 and sum2 == 0.0:
                                label.append(1)
                            else :
                                label.append(2)
                    if args.model == 'cmnist':
                        for j in range(data.shape[0]):
                            dataum = data[j]
                            sum0, sum1, sum2 = dataum[0].sum().item(), dataum[1].sum().item(), dataum[2].sum().item()
                            if sum0 == sum1 and sum1 == sum2:
                                label.append(0)  #white 
                            elif sum1 == 0.0 and sum2 == 0.0:
                                label.append(1)
                            else :
                                label.append(2)


            #label が {g3j, w3t} みたいに出てくる場合
            if args.model =='oscn' or (args.model == 'cmnist_oscn' and target_modality == 1):
                label = list(label)
                
                if n_modality == 0:
                    color_to_int = {
                        'r' : 0,
                        'g' : 1,
                        'w' : 2,
                    }
                    label = [color_to_int[s[0]] for s in label]
                    start_ind, end_ind = 0, 3
                elif n_modality == 1:
                    label = [int(s[1]) for s in label]
                    start_ind, end_ind = 1, 10
                elif n_modality == 2:
                    zukei_to_int = {
                        'j' : 0,
                        's' : 1,
                        't' : 2,
                    }
                    label = [zukei_to_int[s[2]] for s in label]
                    start_ind, end_ind = 0, 3
            #model.reconstruct(data, '.', 3939, n =10)

            """ indr = np.random.randint(0, len(label))
            res = data[indr].cpu().detach().numpy().T
            tar = (res * 255).astype(np.uint8)
            if tar.shape[2] == 1:
                tar = tar[:,:,0]
            pil_image = Image.fromarray(tar) """
            #pil_image.save('check_' + str(label[indr])+ '.png')
            latent_space = model.latent(data)[target_modality].cpu().detach().numpy()
            latent_dim  = latent_space.shape[-1]

            label_all += list(label)
            latent_all += list(latent_space)

            if i == 5:
                break
        print(len(label_all))
        latent_all = np.array(latent_all)
        #latent_all.reshape()
        latent_all = np.reshape(latent_all, (-1, latent_dim))
        label_all = np.array(label_all)
        #np.save('latent_' + args.model + "_" + str(target_modality), latent_all)
        #np.save('label_' + args.model + "_" + str(target_modality), np.array(label_all))
        print("latent_all.shape is : ",latent_all.shape, "len(label_all) is ",len(label_all))
        need_labelchange = False
        xy_all = [[] for i in range(10)]
        
        mean_all = []
        dist_all = [[0.0 for i in range(end_ind - start_ind )] for i in range(end_ind - start_ind )]

       

        for i in range(start_ind, end_ind):
            target_latents = latent_all[np.where(label_all == i)]
            print(target_latents.shape)
            
            mean_all.append(np.mean(target_latents, axis = 0))
            print(mean_all[i-1].shape)
            #print(np.mean(target_latents, axis = 1))
           

        for i in range(start_ind, end_ind):
            for j in range(start_ind, end_ind):
                dist_all[i-1][j-1] = np.linalg.norm(mean_all[i-1] - mean_all[j-1]) 

        for i in range(start_ind - 1, end_ind - 1):
            for j in range(start_ind -1, end_ind - 1):
                print( '{:.1f}'.format(dist_all[i][j]),  end=' ')
            print('')

        if require_2d : #2次元
            points = TSNE(n_components=2, random_state=0,  perplexity = 40).fit_transform(latent_all)
            
            for label, coordinates in zip(label_all, points):
                c_list = coordinates.tolist()
                c0, c1 = c_list[0],  c_list[1]
                if need_labelchange:
                    label_int =  torch.argmax(label, dim=0).item()
                else:
                    label_int = label
                xy_all[label_int].append([ c0, c1 ])
            
            for i in range(start_ind, end_ind):
                result = np.array(xy_all[i])
                plt.scatter(result[:,0], result[:,1], s = 0.5, c = color_dict[i])
            
            plt.savefig('latent_images/latent_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            plt.clf()

    
        if require_3d : #3次元
            points = TSNE(n_components=3, random_state=0).fit_transform(latent_all)
            for label, coordinates in zip(label_all, points):
                c_list = coordinates.tolist()
                c0, c1, c2 = c_list[0], c_list[1], c_list[2]
                if need_labelchange:
                    label_int =  torch.argmax(label, dim=0).item()
                else:
                    label_int = label
                xy_all[label_int].append([ c0, c1, c2 ])

            fig = plt.figure()
            ax = Axes3D(fig)

            for i in range(start_ind, end_ind):
                result = np.array(xy_all[i])
                ax.scatter(result[:,0], result[:,1], result[:, 2], s = 0.5, c = color_dict[i])
            
            plt.savefig('latent_images/3d_latent_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            
        
        