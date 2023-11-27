import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from torch import optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data

matplotlib.use('Agg')

color_dict = {
    0: "gold",
    1: "blue",
    2: "green",
    3: "red",
    4: "aqua",
    5: "#a65628", # brown
    6: "k", # black
    7: 'coral',
    8: 'yellow',
    9: "darkviolet"
}

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

parser.add_argument('--target-modality', type=int, default=0, metavar='M',
                    help='analysis target of information modality (default: 0)')
parser.add_argument('--target-property', type=int, default=1, metavar='P',
                    help='analysis target of information property (default: 1)')
parser.add_argument('--output-dir', type=str, default="final_latent_images_test/", metavar='D',
                    help='save directory of results (default: latent_image)')

# args
args = parser.parse_args()
print('initial args:\n', args)

target_modality_args = args.target_modality
target_property_args = args.target_property
# output_dir = 'final_latent_images_test_231024/qualitative'
output_dir = args.output_dir

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
    print("args are updated using pretrained_path:\n", args)

# args.batch_size = 256  # CHECK
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

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

print('args after model loading:\n', args)


# Functions
def plot_reconstruct():
    model.eval()
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            loss = -t_objective(model, data, K=args.K)
            if i == 0:
                model.reconstruct(data, output_dir, 999, n =64)
                model.generate(output_dir, 999)
                print('Plot reconstruction of index:', i)
                # exit()
                break
    return


def analyze_cluster(latent_all,
                       label_all,
                       target_modality,
                       target_property,
                       ):
    silhouette_vals = silhouette_samples(
        latent_all, label_all, metric='euclidean')
    cluster_labels = np.unique(label_all)
    n_clusters = len(cluster_labels)

    y_ax_lower, y_ax_upper= 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[label_all == c]
        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    print('Silhouette values:', silhouette_vals)
    print('Average of silhouette coef:', silhouette_avg)

    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    plt.savefig(output_dir + '/silhouette_' + args.model + '_' + str(target_modality) + '_' + str(target_property) + '.svg', format='svg')
    plt.clf()

    return {
        'cluster_avg': silhouette_avg,
        'cluster_all': silhouette_vals,
    }


def analyze_magnitude(label_all,
                 latent_all,
                 target_property,
                 category_num,
                 start_ind,
                 end_ind,
                 withzero):
    mean_all = [None for i in range(category_num)]
    dist_all = [[0.0 for i in range(category_num)] for i in range(category_num)]

    for i in range(start_ind, end_ind):
        target_latents = latent_all[np.where(label_all == i)]
        # print(target_latents.shape)

        mean_all[i - start_ind] = np.mean(target_latents, axis = 0)
        # print(mean_all[i-1].shape)
        # print(np.mean(target_latents, axis = 1))

    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            dist_all[i - start_ind][j - start_ind] = \
                np.linalg.norm(mean_all[i - start_ind] - mean_all[j - start_ind])

    if target_property == 1:
        if args.model == 'mnist_clevr' or args.model == 'clevr':
            print('  3   4   5   6   7   8   9   10')
        elif withzero:
            print('  0   1   2   3   4   5   6   7   8   9')
        else:
            print('  1   2   3   4   5   6   7   8   9')
    elif target_property == 0:
        print('  0   1   2   3')
    elif target_property == 2:
        print('  0   1   2')

    for i in range(start_ind, end_ind):
        print(i , end = ' ')
        for j in range(start_ind , end_ind ):
            print( '{:.1f}'.format(dist_all[i - start_ind][j - start_ind]),  end=' ')
        print('')

    dist_flat = []
    dist_cor1, dist_cor2 = [], []
    for i in range(start_ind , end_ind):
        for j in range(i , end_ind ):
            dist_flat.append([abs(i-j), dist_all[i - start_ind][j - start_ind]])
            if i != j:
                dist_cor1.append(abs(i-j))
                dist_cor2.append(dist_all[i - start_ind][j - start_ind])
    dist_flat = np.array(dist_flat)

    coef = np.corrcoef([dist_cor1, dist_cor2])
    correlation = coef[0, 0]
    print('correlation', coef)

    plt.scatter(dist_cor1, dist_cor2)
    plt.savefig(output_dir + '/dist_relations.svg', format='svg')
    plt.clf()

    return {
        'magnitude_avg:' correlation,
        'magnitude_all:' correlation,
    }


def analyze_tsne_2d(label_all,
                    latent_all,
                    target_modality,
                    target_property,
                    category_num,
                    start_ind,
                    end_ind
                    ):
    points = TSNE(
        n_components=2,
        random_state=0,
        perplexity=20
    ).fit_transform(latent_all)
    # points = TSNE(n_components=2, random_state=0).fit_transform(latent_all)

    xy_all = [[] for i in range(category_num)]
    print(xy_all, label_all, points)

    fig, ax = plt.subplots()

    for label, coordinates in zip(label_all, points):
        c_list = coordinates.tolist()
        c0, c1 = c_list[0],  c_list[1]
        xy_all[int((label - start_ind).item())].append([ c0, c1 ])

    for i in range(start_ind, end_ind):
        result = np.array(xy_all[i - start_ind])
        ax.scatter(result[:, 0], result[:, 1],
                   s=0.5, c=color_dict[i - start_ind])

        if target_property == 1:
            ax.scatter(
                [], [], label=str(i), s=2, c=color_dict[i- start_ind]) # 凡例Aのダミープロット
        elif target_property == 0:
            coldic = ["white", "blue", 'green', 'red']
            ax.scatter(
                [], [], label=coldic[i], s=2, c=color_dict[i-start_ind]) # 凡例Aのダミープロット
        elif target_property == 2:
            coldic = ["cross", "square", 'triangle']
            ax.scatter(
                [], [], label=coldic[i], s=2, c=color_dict[i - start_ind])

        # for a, b in zip(result[:,0][:100], result[:,1][:100]):
        #     print(a,b)
        #     plt.text(a, b, str(i), c = "black", fontsize=6, clip_on=True)

    if target_property == 1:
        ax.legend(loc='lower right', ncol=2, handletextpad = 0.4, borderpad = 0.2)
    else:
        ax.legend(loc='lower right',  handletextpad = 0.4, borderpad = 0.2)

    fig.savefig(output_dir + '/latent_' + args.model + '_' + str(target_modality) + '_' + str(target_property) + '.svg', format='svg')
    print('saved ' + output_dir + '/latent_' + args.model + '_' + str(target_modality) + '_' + str(target_property) + '.svg')
    plt.clf()

    return


def analyze_tsne_3d(label_all,
                    latent_all,
                    category_num,
                    target_modality,
                    target_property,
                    need_labelchange = False
                    ):
    # points = TSNE(n_components=3, random_state=0, perplexity = 30).fit_transform(latent_all)
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
        ax.scatter(result[:,0], result[:,1], result[:, 2],
                   s = 0.5, c = color_dict[i - start_ind])

    plt.savefig(output_dir + '/3d_latent_' + args.model + '_' + str(target_modality) + '_' + str(target_property) + '.svg', format='svg')
    return


def analyze_additive(latent_all,
                     label_all,
                     target_modality,
                     category_num,
                     start_ind,
                     end_ind
                     ):
    mean_all = [None for i in range(category_num) ]
    dist_all = [[0.0 for i in range(category_num)] for i in range(category_num)]

    for i in range(start_ind, end_ind):
        target_latents = latent_all[np.where(label_all == i)]
        mean_all[i] = np.mean(target_latents, axis = 0)
        # print(np.mean(target_latents, axis = 1))
        print('shape', mean_all[i].shape)

    # mean_all_for_calc = np.array(mean_all[start_ind:])
    # model.generate_special(mean_all[0])
    # base = mean_all_for_calc.mean(axis = 0)
    # base[8] = (mean_all[1] + mean_all[8]  - mean_all[4])[8]
    # base[8] = (mean_all[0] + mean_all[7] - mean_all[4])[8]
    # model.generate_special( base  )

    # if False :
    if True:
        # TODO: fix try to if
        try:
            model.generate_special(
                mean_all[1] + mean_all[9] - mean_all[8],
                target_modality = target_modality,
                label = "1+9-8")
        except:
            model.generate_special(
                mean = mean_all[1] + mean_all[9] - mean_all[8],
                label = "1+9-8" )
        try:
            model.generate_special(
                mean_all[2] + mean_all[7] - mean_all[1],
                target_modality = target_modality,
                label = "2+7-1")
        except:
            model.generate_special(
                mean = mean_all[2] + mean_all[7] - mean_all[1],
                label = "2+7-1" )

        try:
            model.generate_special(
                mean_all[3] + mean_all[5] - mean_all[2],
                target_modality = target_modality,
                label = "3+5-2")
        except:
            model.generate_special(
                mean = mean_all[3] + mean_all[5] - mean_all[2],
                label = "3+5-2" )

        """
        points = TSNE(n_components=3, random_state=0).fit_transform(mean_all)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(points[:,0], points[:,1], points[:, 2], s = 0.5, c = color_dict[i])
        """

    if True:
        for i in range(start_ind, end_ind):
            for k in range(mean_all[1].shape[0]):
                v = mean_all[i][k]
                if v < 0:
                    print('{:.1f}'.format(v), end = " ")
                else:
                    print(' ',end = '')
                    print('{:.1f}'.format(v), end = " ")
            print("")
            # print(mean_all.mean(axis = 0))
        plt.savefig(output_dir + '/special_latent.svg', format='svg')

    if True:
        check_additive(1, 9, 8, mean_all=mean_all)
        check_additive(2, 7, 1, mean_all=mean_all)
        check_additive(3, 5, 2, mean_all=mean_all)
        # check_additive(0, 3, 2, mean_all=mean_all) 0 is error
        check_additive(1, 8, 4, mean_all=mean_all)

    return


def check_additive(n1, n2 , n3, mean_all):
    """Performn calculation of `n1 + n2 - n3`
    """

    result = mean_all[n1] + mean_all[n2] - mean_all[n3]
    target = mean_all[n1 + n2 - n3]

    res_all = []
    min_dist, min_ind = 1e9, -1
    # for i in range(9):  # original
    for i in range(1, 10):
        print(type(mean_all[i]))
        dist = np.linalg.norm(mean_all[i] - result)
        print(i, dist)
        if dist < min_dist:
            min_ind = i
            min_dist = dist
    print(min_ind, n1 + n2 -n3)
    return


def get_latent_space(
        target_modality,
        target_property,
        category_num):

    label_all = []
    latent_all = []

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

    # Settings for label and latent states
    for i, dataT in enumerate(train_loader):
        data, label = unpack_data(
            dataT,
            device=device,
            require_label=True,
            target=target_modality
        )

        # check data structure and content
        if i == 0:
            print(
                '\n===\nData and label info...',
                '\ntype(label):', type(label),
                '\nlabel:', label,
                '\nlabel[0]:', label[0],
                '\nlabel[1]:', label[1],
                '\ntype(data):', type(data),
                '\ndata[0]:', data[0],
                '\ndata[0].shape:', data[0].shape
            )

        if args.model == 'cmnist_oscn':
            if target_property == 1:
                label = label[0]
            else:
                label = label[1]  # 全部の情報が必要

        # Pattern of label -> [0,1,3,,,] (CMNIST and CLEVR)
        if (args.model == 'cmnist') or (args.model == 'cmnist_oscn' and target_property == 1) or (args.model == 'clevr') or (args.model == 'mnist_clevr'):
             if target_property == 0:
                label = []

                if args.model == 'cmnist_oscn' :
                    for j in range(data[0].shape[0]):
                        dataum = data[0][j]
                        (sum0, sum1, sum2) = \
                            (dataum[0].sum().item(),
                             dataum[1].sum().item(),
                             dataum[2].sum().item())
                        if sum1 == 0.0 and sum2 == 0.0: # R
                            label.append(0)
                        elif sum1 == 0.0 and sum0 == 0.0: # B
                            label.append(1)
                        elif sum2 == 0.0 and sum0 == 0.0: # G
                            label.append(2)
                        else:
                            label.append(3)
                if args.model == 'cmnist':
                    for j in range(data.shape[0]):
                        dataum = data[j]
                        (sum0, sum1, sum2) = \
                            (dataum[0].sum().item(),
                             dataum[1].sum().item(),
                             dataum[2].sum().item())
                        if sum1 == 0.0 and sum2 == 0.0: # B
                            label.append(0)
                        elif sum1 == 0.0 and sum0 == 0.0: # B
                            label.append(1)
                        elif sum2 == 0.0 and sum0 == 0.0: # B
                            label.append(2)
                        else:
                            label.append(3)

        # Pattern of label -> (g3j, w3t) (OSCN)
        elif (args.model =='oscn') or (args.model == 'cmnist_oscn' and target_property != 1):
            label = list(label)
            if target_property == 0:
                try :
                    label = [color_to_int[s[0]] for s in label]
                except:
                    pass
            elif target_property == 1:
                try :
                    label = [int(s[1]) for s in label]
                except:
                    pass
            elif target_property == 2:
                try:
                    label = [zukei_to_int[s[2]] for s in label]
                except:
                    pass
            else:
                raise Exception
        else:
            raise Exception

        # Get latent space
        # model.reconstruct(data, '.', 3939, n =10)
        # pil_image.save(output_dir + '/check_' + str(label[indr])+ '.png')
        latent_space = model.latent(data)[target_modality].cpu().detach().numpy()
        latent_dim  = latent_space.shape[-1]
        if args.model == 'mnist_clevr':
            label = label[0]
        try :
            label_all += list(np.array(label.cpu()))
        except :
            label_all += list(np.array(label))
        latent_all += list(latent_space)
        if i == 15:  # CHECK
            break

    latent_all = np.array(latent_all)
    latent_all = np.reshape(latent_all, (-1, latent_dim))
    label_all = np.array(label_all)

    return (latent_all, label_all)


def get_index(
        target_modality,
        target_property,
        withzero):
    # OSCN_CMNIST
    if target_property == 0:
        start_ind, end_ind = 0, 4
    elif target_property == 1:
        start_ind, end_ind = 1, 10
        if withzero:
            start_ind, end_ind = 0, 10
    elif target_property == 2:
        start_ind, end_ind = 0, 3
    else:
        raise Exception

    # CLEVR
    if args.model == 'clevr':
        start_ind, end_ind = 3, 11
    elif args.model == 'mnist_clevr':
        start_ind, end_ind = 3, 10

    return (start_ind, end_ind)


def analyze(target_modality,
            target_property):

    # What analysis are performed?
    if args.model == 'cmnist_oscn':
        require_reconstruct = True
    else:
        require_reconstruct = False
    require_magnitude = True
    require_2d = True
    require_3d = False  # if true, error happened.
    require_cluster = True
    if target_property == 1:
        require_additive = True
    elif target_property == 0 or target_property == 2:
        require_additive = False
    else:
        raise Exception
    withzero = False

    # Settings for the number of cateogry
    if target_property == 0:
        category_num = 4
    elif target_property == 1:
        category_num = 10
    elif target_property == 2:
        category_num = 3
        if args.model == 'clevr' or args.model == 'mnist_clevr':
            category_num = 8
    else:
        raise Exception

    # Start and end index of category number
    start_ind, end_ind =  get_index(
        target_modality=target_modality,
        target_property=target_property,
        withzero=withzero)

    # Get latent space
    latent_all, label_all = get_latent_space(
        target_modality=target_modality,
        target_property=target_property,
        category_num=category_num,
    )

    # np.save('latent_' + args.model + "_" + str(target_modality), latent_all)
    # np.save('label_' + args.model + "_" + str(target_modality), np.array(label_all))

    print(
        '\n===\nSetting info...',
        '\nTarget modality:', target_modality,
        '\nTarget property:', target_property,
        '\nNumber of category', category_num,
        '\nShape of latent_all:', latent_all.shape,
        '\nShape of label_all: ', label_all.shape,
        '\nSet of label_all: ', set(label_all),
        '\nStart index:', start_ind,
        '\nEnd index:', end_ind,
        '\nwithzero:', withzero
    )

    # Main analysis
    if require_reconstruct:
        plot_reconstruct()
    if require_cluster:
        analyze_cluster(
            latent_all=latent_all,
            label_all=label_all,
            target_modality=target_modality,
            target_property=target_property,
        )
    if require_magnitude:
        analyze_magnitude(
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_property=target_property,
            start_ind=start_ind,
            end_ind=end_ind,
            withzero=withzero,
        )
    if require_2d:
        analyze_tsne_2d(
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_modality=target_modality,
            target_property=target_property,
            start_ind=start_ind,
            end_ind=end_ind,
        )
    if require_3d:
        analyze_tsne_3d(
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_modality=target_modality,
            target_property=target_property,
        )
    if require_additive:
        analyze_additive(
            latent_all=latent_all,
            label_all=label_all,
            target_modality=target_modality,
            category_num=category_num,
            start_ind=start_ind,
            end_ind=end_ind,
        )
    return


def visualize_latent():
    """
    target_modality
        Whether latent states are analyzed
        Only Single modal (OSCN or CMNIST) -> 0
        CMNIST_OSCN -> cmnist (0) or oscn(1)?
    target_property
        Whether information are analyzed, color(0), number(1), shape(2)?
    """
    target_modality = target_modality_args
    target_property = target_property_args
    if args.model == 'cmnist' or args.model == 'oscn':
        target_modality = 0

    print('\n\nSTART ANALYSIS')
    analyze(
        target_modality=target_modality,
        target_property=target_property,
    )
    return


if __name__ == '__main__':
    agg = defaultdict(list)
    model.eval()
    visualize_latent()
