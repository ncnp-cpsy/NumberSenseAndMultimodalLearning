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
from sklearn.metrics import silhouette_samples
from matplotlib import cm

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data

matplotlib.use('Agg')

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

print(args.model)
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
    if not True:
        model.eval()
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                data = unpack_data(dataT, device=device)
                loss = -t_objective(model, data, K=args.K)
                if i == 0:
                    model.reconstruct(data, '.', 999, n =64)
                    model.generate('.', 999)
                    exit()
        
    if True:
        color_dict = {
            0 : "gold",
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

        color_to_int = {
            'r' : 3,
            'g' : 2,
            'b' : 1,
            'w' : 0,
        }

        zukei_to_int = {
            'j' : 0,
            's' : 1,
            't' : 2,
        }



        agg = defaultdict(list)
        model.eval()
        
        label_all = []
        latent_all = []
        # cmnist か oscn か、ということ。0ならcmnist, 1ならoscn。どっちの潜在空間が気になる？
        target_modality = 1

        if args.model != 'cmnist_oscn' and args.model != 'mnist_clevr' :
            target_modality = 0
        
        # その中で 0 : 色、　1 : 数、 2: 形
        n_modality = 1
        
        require_dist = True
        require_2d = True
        require_3d = False
        require_additive = False
        require_silhouette = True
        output_dir = 'final_latent_images_test'
        withzero = False


        #モダリティによってカテゴリ数を調節
        if n_modality == 1:
            category_num = 10
        elif n_modality == 0:
            category_num = 4
        else:
            category_num = 3
            if args.model == 'clevr' or args.model == 'mnist_clevr':
                category_num = 8


        #モダリティによってインデックスが変わる
        start_ind, end_ind = 1, 10
        if withzero:
            start_ind, end_ind = 0, 10

        for i, dataT in enumerate(train_loader):
            data,label = unpack_data(dataT, device=device, require_label = True, target = target_modality)
            
            if args.model == 'cmnist_oscn':
                if n_modality == 1:
                    label = label[0] 
                else:
                    label = label[1]#全部の情報が必要

            #label が [0,1,3,,,]みたいに出てくる場合
            if (args.model == 'cmnist') or (args.model == 'cmnist_oscn' and n_modality == 1) or (args.model == 'clevr') or (args.model == 'mnist_clevr'):
                if n_modality == 0:
                    label = []
                    start_ind, end_ind = 0, 4

                    if args.model == 'cmnist_oscn' : 
                        for j in range(data[0].shape[0]):
                            dataum = data[0][j]
                            sum0, sum1, sum2 = dataum[0].sum().item(), dataum[1].sum().item(), dataum[2].sum().item()
                            if sum1 == 0.0 and sum2 == 0.0: #R
                                label.append(0)
                            elif sum1 == 0.0 and sum0 == 0.0: #B
                                label.append(1) 
                            elif sum2 == 0.0 and sum0 == 0.0: #G
                                label.append(2) 
                            else:
                                label.append(3)
 
                    if args.model == 'cmnist':
                        for j in range(data.shape[0]):
                            dataum = data[j]
                            sum0, sum1, sum2 = dataum[0].sum().item(), dataum[1].sum().item(), dataum[2].sum().item()
                            if sum1 == 0.0 and sum2 == 0.0: #B
                                label.append(0)
                            elif sum1 == 0.0 and sum0 == 0.0: #B
                                label.append(1) 
                            elif sum2 == 0.0 and sum0 == 0.0: #B
                                label.append(2) 
                            else:
                                label.append(3)
            #label が {g3j, w3t} みたいに出てくる場合
            if (args.model =='oscn') or (args.model == 'cmnist_oscn' and n_modality != 1):
                label = list(label)
                if n_modality == 0:
                    try :
                        label = [color_to_int[s[0]] for s in label]
                    except:
                        pass
                    start_ind, end_ind = 0, 4
                elif n_modality == 1:
                    try :
                        label = [int(s[1]) for s in label]
                    except:
                        pass
                    start_ind, end_ind = 1, 10
                    if withzero:
                        start_ind, end_ind = 0, 10
                elif n_modality == 2:
                    try:
                        label = [zukei_to_int[s[2]] for s in label]
                    except:
                        pass
                    start_ind, end_ind = 0, 3
            #model.reconstruct(data, '.', 3939, n =10)
            #pil_image.save('check_' + str(label[indr])+ '.png')
            latent_space = model.latent(data)[target_modality].cpu().detach().numpy()
            latent_dim  = latent_space.shape[-1]
            if args.model == 'mnist_clevr':
                label = label[0]
            try : 
                label_all += list(np.array(label.cpu()))
            except :
                label_all += list(np.array(label))
            latent_all += list(latent_space)
            if i == 15:
                break

        if args.model == 'clevr': 
            start_ind, end_ind = 3, 11
        elif args.model == 'mnist_clevr':
            start_ind, end_ind = 3, 10

        print(len(label_all))
        latent_all = np.array(latent_all)
        #latent_all.reshape()
        latent_all = np.reshape(latent_all, (-1, latent_dim))
        label_all = np.array(label_all)
        #np.save('latent_' + args.model + "_" + str(target_modality), latent_all)
        #np.save('label_' + args.model + "_" + str(target_modality), np.array(label_all))
        print("latent_all.shape is : ",latent_all.shape, "len(label_all) is ",len(label_all))
        need_labelchange = False
        
        if require_silhouette:
            #points = TSNE(n_components=2, random_state=0).fit_transform(latent_all)
            #latent_all = points
            silhouette_vals = silhouette_samples(latent_all, label_all, metric='euclidean')
            cluster_labels = np.unique(label_all)     

            y_ax_lower, y_ax_upper= 0,0
            yticks = []
            n_clusters = len(cluster_labels)

            for i,c in enumerate(cluster_labels):
                c_silhouette_vals = silhouette_vals[label_all == c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
                c_silhouette_vals.sort()
                y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
                color = cm.jet(float(i)/n_clusters)               # 色の値を作る
                plt.barh(range(y_ax_lower, y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                                c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                                height=1.0,                      # 棒の高さ
                                edgecolor='none',                # 棒の端の色
                                color=color)                     # 棒の色
                yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
                y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

            silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
            print('sil:', silhouette_avg)
            plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
            plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
            plt.ylabel('Cluster')
            plt.xlabel('silhouette coefficient')
            plt.savefig(output_dir + '/silhouette_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            plt.clf()

        if require_dist :    
            mean_all = [None for i in range(category_num)]
            dist_all = [[0.0 for i in range(category_num )] for i in range(category_num )]
            
            for i in range(start_ind, end_ind):
    
                target_latents = latent_all[np.where(label_all == i)]
                #print(target_latents.shape)

                mean_all[i - start_ind] = np.mean(target_latents, axis = 0)
                #print(mean_all[i-1].shape)
                #print(np.mean(target_latents, axis = 1))
            
            
            for i in range(start_ind, end_ind):
                for j in range(start_ind, end_ind):
                    dist_all[i - start_ind][j - start_ind] = np.linalg.norm(mean_all[i - start_ind] - mean_all[j - start_ind]) 
            if n_modality == 1:
                if args.model == 'mnist_clevr' or args.model == 'clevr':
                    print('  3   4   5   6   7   8   9   10')
                elif withzero:
                    print('  0   1   2   3   4   5   6   7   8   9')
                else:
                    print('  1   2   3   4   5   6   7   8   9')
            elif n_modality == 0:
                print('  0   1   2   3')
            elif n_modality == 2:
                print('  0   1   2')

            for i in range(start_ind, end_ind ):
                
                print(i , end = ' ')
                for j in range(start_ind , end_ind ):
                    print( '{:.1f}'.format(dist_all[i - start_ind][j - start_ind]),  end=' ')
                print('')

            dist_flat = []
            dist_cor1, dist_cor2 = [], []
            for i in range(start_ind , end_ind ):
                for j in range(i , end_ind ):
                    dist_flat.append([abs(i-j), dist_all[i - start_ind][j - start_ind]])
                    if i != j:
                        dist_cor1.append(abs(i-j))
                        dist_cor2.append(dist_all[i - start_ind][j - start_ind])
            dist_flat = np.array(dist_flat)
            print(np.corrcoef([dist_cor1, dist_cor2]))
            plt.scatter(dist_cor1, dist_cor2)
            plt.savefig('dist_relations.png')
            plt.clf()

        if require_2d : #2次元
            points = TSNE(n_components=2, random_state=0,  perplexity = 20).fit_transform(latent_all)
            #points = TSNE(n_components=2, random_state=0).fit_transform(latent_all)

            xy_all = [[] for i in range(category_num)] 

            fig, ax = plt.subplots()

            for label, coordinates in zip(label_all, points):
                c_list = coordinates.tolist()
                c0, c1 = c_list[0],  c_list[1]
                xy_all[int((label - start_ind).item())].append([ c0, c1 ])
            
            for i in range(start_ind, end_ind):
                result = np.array(xy_all[i - start_ind])
                ax.scatter(result[:,0], result[:,1], s = 0.5, c = color_dict[i- start_ind])
                if n_modality == 1:
                    ax.scatter([], [], label=str(i), s = 2, c = color_dict[i- start_ind]) #凡例Aのダミープロット
                elif n_modality == 0:
                    coldic = ["white", "blue", 'green', 'red']
                    ax.scatter([], [], label=coldic[i], s = 2, c = color_dict[i- start_ind]) #凡例Aのダミープロット
                elif n_modality == 2:
                    coldic = ["cross", "square", 'triangle']
                    ax.scatter([], [], label=coldic[i], s = 2, c = color_dict[i- start_ind]) 

                #for a, b in zip(result[:,0][:100], result[:,1][:100]):
                    #print(a,b)
                    #plt.text(a, b, str(i), c = "black", fontsize=6, clip_on=True)

            if n_modality == 1:
                ax.legend(loc='lower right', ncol=2, handletextpad = 0.4, borderpad = 0.2)
            else:
                ax.legend(loc='lower right',  handletextpad = 0.4, borderpad = 0.2)
            fig.savefig(output_dir + '/latent_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            print('saved ' + output_dir + '/latent_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            plt.clf()


    
        if require_3d : #3次元
            #points = TSNE(n_components=3, random_state=0, perplexity = 30).fit_transform(latent_all)
            points = TSNE(n_components=3, random_state=0).fit_transform(latent_all)
            xy_all = [[] for i in range(10)]
            for label, coordinates in zip(label_all, points):
                c_list = coordinates.tolist()
                c0, c1, c2 = c_list[0], c_list[1], c_list[2]
                if need_labelchange:
                    label_int =  torch.argmax(label, dim=0).item()
                else:
                    label_int = label
                xy_all[int((label_int - start_ind).item())].append([ c0, c1, c2 ])
            fig = plt.figure()
            ax = Axes3D(fig)

            for i in range(start_ind, end_ind):
                result = np.array(xy_all[i - start_ind])
                ax.scatter(result[:,0], result[:,1], result[:, 2], s = 0.5, c = color_dict[i - start_ind])
            
            plt.savefig(output_dir + '/3d_latent_' + args.model + '_' + str(target_modality) + '_' + str(n_modality) + '.png')
            
        
        if require_additive :
            mean_all = [None for i in range(category_num) ]
            dist_all = [[0.0 for i in range(category_num)] for i in range(category_num)]

            for i in range(start_ind, end_ind):
                target_latents = latent_all[np.where(label_all == i)]
                
                mean_all[i] = np.mean(target_latents, axis = 0)
                #print(np.mean(target_latents, axis = 1))
            #mean_all_for_calc = np.array(mean_all[start_ind:])
            #model.generate_special(mean_all[0])
            #base = mean_all_for_calc.mean(axis = 0)
            #base[8] = (mean_all[1] + mean_all[8]  - mean_all[4])[8]
            #base[8] = (mean_all[0] + mean_all[7] - mean_all[4])[8]
            #model.generate_special( base  ) 

            if False :
                try:
                    model.generate_special( mean_all[1] +mean_all[9] - mean_all[8], target_modality = target_modality, label = "1+9-8")
                except:
                    model.generate_special(mean = mean_all[1] +mean_all[9] - mean_all[8],label = "1+9-8" )

                try:
                    model.generate_special( mean_all[2] +mean_all[7] - mean_all[1], target_modality = target_modality, label = "2+7-1")
                except:
                    model.generate_special(mean = mean_all[2] +mean_all[7] - mean_all[1],label = "2+7-1" )

                try:
                    model.generate_special( mean_all[3] +mean_all[5] - mean_all[2], target_modality = target_modality, label = "3+5-2")
                except:
                    model.generate_special(mean = mean_all[3] +mean_all[5] - mean_all[2],label = "3+5-2" )


            """ points = TSNE(n_components=3, random_state=0).fit_transform(mean_all)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(points[:,0], points[:,1], points[:, 2], s = 0.5, c = color_dict[i]) """

            if not True:
                for i in range(start_ind, end_ind):
                    for k in range(mean_all[1].shape[0]):
                        v = mean_all[i][k]
                        if v < 0:
                            print('{:.1f}'.format(v), end = " ")
                        else:
                            print(' ',end = '')
                            print('{:.1f}'.format(v), end = " ")
                    print("")
                #print(mean_all.mean(axis = 0))

                plt.savefig('special_latent.png')
            if False:
                def check_additive(n1, n2 , n3):
                    #n1 + n2 - n3をやる
                    result = mean_all[n1] + mean_all[n2] - mean_all[n3]
                    target = mean_all[n1+n2-n3]

                    res_all = []
                    min_dist, min_ind = 1e9, -1
                    for i in range(9):
                        dist = np.linalg.norm(mean_all[i] - result) 
                        print(i, dist)
                        if dist < min_dist:
                            min_ind = i
                            min_dist = dist
                    print(min_ind, n1 + n2 -n3)
                    return 
                check_additive(0,3,2)
                check_additive(1,8,4)
