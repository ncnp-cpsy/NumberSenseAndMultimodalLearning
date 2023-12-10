import sys
import itertools
import csv
import pprint
from pathlib import Path

import numpy as np
import torch
from torch import optim
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_samples,
)
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from src.utils import unpack_data, Logger
from src.runner import Runner
from src.datasets import convert_label_to_int

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

def analyse_classifier(classifier,
                       output_dir='./',
                       ):
    return {
        'classifier_avg': np.nan,
        'classifier_all': np.nan,
    }

def count_reconst(recon,
                  true_label,
                  classifier,
                  output_dir='./',
                  suffix='',
                  set_label=None
                  ):
    def accuracy(pred, tar):
        print('pred:', pred, '\ntar:', tar)
        acc = torch.sum(pred == tar)
        return acc, acc / len(pred)
    pred_label = classifier.predict(recon)
    pred_label = torch.argmax(pred_label, dim=1).cpu() + 1
    true_label = true_label.cpu()
    if set_label is None:
        set_label = list(set(true_label.numpy()))

    # all category
    acc_all, acc_ratio = accuracy(
        pred=pred_label,
        tar=true_label,
    )
    print('Accuracy (count):', acc_all, '\nAccuracy (ratio)', acc_ratio)

    # each cateogry
    conf_mat = confusion_matrix(
        y_true=true_label,
        y_pred=pred_label,
        labels=set_label,
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat,
        display_labels=set_label,
    )
    fname = output_dir + '/confusion' + suffix + '.svg'
    disp.plot()
    plt.savefig(fname, format='svg')
    plt.clf()
    print('Accuracy:\n', conf_mat)

    # normalized
    conf_mat_nrm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat_nrm,
        display_labels=set_label,
    )
    fname = output_dir + '/confusion_normalized' + suffix + '.svg'
    disp.plot()
    plt.savefig(fname, format='svg')
    plt.clf()
    print('Accuracy:\n', np.round(conf_mat_nrm, decimals=3))

    return acc_ratio.item(), conf_mat_nrm

def analyse_reconst(runner,
                    classifier,
                    target_modality=None,
                    output_dir='./',
                    ):
    runner.model.eval()
    with torch.no_grad():
        for i, dataT in enumerate(runner.test_loader):
            data, label = unpack_data(
                dataT,
                device=runner.args.device,
                require_label=True,
            )
            # loss = - runner.t_objective(runner.model, data, K=runner.args.K)
            if i == 0:
                recon = runner.model.reconstruct(
                    data=data,
                    output_dir=output_dir,
                    suffix=999,
                )
                if 'MMVAE' in runner.model_name:
                    print('size of extracted recon in MMVAE:', type(recon))
                    recon = recon[target_modality][target_modality]
                    print('size of extracted recon in MMVAE:', recon.shape)
                    recon = recon[0]
                    print('size of extracted recon in MMVAE:', recon.shape)
                generation = runner.model.generate(
                    output_dir=output_dir,
                    suffix=999,
                )
                acc, confusion = count_reconst(
                    recon=recon,
                    true_label=convert_label_to_int(
                        label=label,
                        model_name=runner.model_name,
                        target_property=1,
                        target_modality=target_modality,
                        data=data,
                    ),
                    classifier=classifier,
                    output_dir=output_dir,
                    suffix='_reconst',
                )
                break

    suffix = str(target_modality) + 'x' + str(target_modality)
    return {
        'reconst_' + suffix + '_avg': acc,
        'reconst_' + suffix + '_all': np.nan,
    }

def analyse_cross(runner,
                  classifier,
                  target_modality_from=0,
                  target_modality_to=1,
                  output_dir='./',
                  ):
    runner.model.eval()
    with torch.no_grad():
        for i, dataT in enumerate(runner.test_loader):
            data, label = unpack_data(
                dataT,
                device=runner.args.device,
                require_label=True,
            )
            if i == 0:
                recon = runner.model.reconstruct(
                    data=data,
                    output_dir=output_dir,
                    suffix=999,
                )
                if 'MMVAE' in runner.model_name:
                    print('size of extracted recon in MMVAE:', type(recon))
                    recon = recon[target_modality_from][target_modality_to]
                    print('size of extracted recon in MMVAE:', recon.shape)
                    recon = recon[0]
                    print('size of extracted recon in MMVAE:', recon.shape)
                acc, confusion = count_reconst(
                    recon=recon,
                    true_label=convert_label_to_int(
                        label=label,  # Not label[target_modality_to]
                        model_name=runner.model_name,
                        target_property=1,
                        target_modality=target_modality_to,
                        data=data,
                    ),
                    classifier=classifier,
                    output_dir=output_dir,
                    suffix='_cross',
                )
                break

    suffix = str(target_modality_from) + 'x' + str(target_modality_to)
    return {
        'cross_' + suffix + '_avg': acc,
        'cross_' + suffix + '_all': np.nan,
    }

def analyse_cluster(latent_all,
                    label_all,
                    target_modality,
                    target_property,
                    output_dir='./',
                    ):
    silhouette_vals = silhouette_samples(
        latent_all, label_all, metric='euclidean')
    cluster_labels = np.unique(label_all)
    n_clusters = len(cluster_labels)

    y_ax_lower, y_ax_upper = 0, 0
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
    fname = output_dir + '/silhouette_' + str(target_modality) + '_' + str(target_property) + '.svg'
    plt.savefig(fname, format='svg')
    plt.clf()

    return {
        'cluster_avg': silhouette_avg,
        'cluster_all': silhouette_vals,
    }


def analyse_magnitude(runner,
                      label_all,
                      latent_all,
                      target_property,
                      category_num,
                      start_ind,
                      end_ind,
                      withzero,
                      trans_log=False,
                      output_dir='./',
                      ):

    # latent embeddings for each number (dim: 9)
    mean_all = [None for i in range(category_num)]
    # distance between numbers (dim: 9 x 9)
    dist_all = [[0.0 for i in range(category_num)] for i in range(category_num)]

    for i in range(start_ind, end_ind):
        target_latents = latent_all[np.where(label_all == i)]
        mean = np.mean(target_latents, axis = 0)
        if trans_log:
            print('Perform log transformation')
            print('Before:\n', mean)
            mean = np.exp(mean)
            print('After:\n', mean)
        mean_all[i - start_ind] = mean
        # print(target_latents.shape)
        # print(mean_all[i-1].shape)
        # print(np.mean(target_latents, axis = 1))

    for i in range(start_ind, end_ind):
        for j in range(start_ind, end_ind):
            dist_all[i - start_ind][j - start_ind] = \
                np.linalg.norm(mean_all[i - start_ind] - mean_all[j - start_ind])

    print('\nDistance matrix between labels.')
    if target_property == 0:
        print('  0   1   2   3')
    elif target_property == 1:
        if runner.model_name == 'MMVAE_MNIST_CLEVR' or runner.model_name == 'VAE_CLEVR':
            print('  3   4   5   6   7   8   9   10')
        elif withzero:
            print('  0   1   2   3   4   5   6   7   8   9')
        else:
            print('  1   2   3   4   5   6   7   8   9')
    elif target_property == 2:
        print('  0   1   2')
    else:
        Exception

    for i in range(start_ind, end_ind):
        print(i , end = ' ')
        for j in range(start_ind , end_ind ):
            print( '{:.1f}'.format(dist_all[i - start_ind][j - start_ind]),  end=' ')
        print('')

    # Make matrix representing distances of class and latent embedding
    dist_flat = []
    dist_cor1, dist_cor2 = [], []
    for i in range(start_ind, end_ind):
        for j in range(i, end_ind):
            dist_flat.append([abs(i-j), dist_all[i - start_ind][j - start_ind]])
            if i != j:
                # distance of number (label)
                dist_cor1.append(abs(i-j))
                # distance of latent embedding
                dist_cor2.append(dist_all[i - start_ind][j - start_ind])
    dist_flat = np.array(dist_flat)
    print(
        '\nDistance of number label and latent embedding',
        '\nNumber:', dist_cor1,
        '\nLatent embbeding:', dist_cor2,
        '\nFlatten array:', dist_flat,
    )
    plt.scatter(dist_cor1, dist_cor2)
    plt.savefig(output_dir + '/dist_relations.svg', format='svg')
    plt.xlabel('Class distance')
    plt.ylabel('Distance of latent embeddings')
    plt.clf()

    # Correlation
    coef = np.corrcoef([dist_cor1, dist_cor2])
    correlation = coef[0, 1]
    print('Correlation:', coef)

    return {
        'magnitude_avg': np.abs(correlation),
        'magnitude_all': correlation,
    }


def analyse_tsne_2d(label_all,
                    latent_all,
                    target_modality,
                    target_property,
                    category_num,
                    start_ind,
                    end_ind,
                    output_dir='./',
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

    fig.savefig(output_dir + '/latent_' + '_' + str(target_modality) + '_' + str(target_property) + '.svg', format='svg')
    print('saved ' + output_dir + '/latent_' + '_' + str(target_modality) + '_' + str(target_property) + '.svg')
    plt.clf()

    return {
        'tsne-2d_avg': np.nan,
        'tsne-2d_all': np.nan,
    }


def analyse_tsne_3d(label_all,
                    latent_all,
                    category_num,
                    target_modality,
                    target_property,
                    need_labelchange = False,
                    output_dir='./',
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

    fname = output_dir + '/3d_latent_' + str(target_modality) + '_' + str(target_property) + '.svg'
    plt.savefig(fname, format='svg')
    plt.clf()
    return {
        'tsne-3d_avg': np.nan,
        'tsne-3d_all': np.nan,
    }


def analyse_mathematics(runner,
                        classifier,
                        latent_all,
                        label_all,
                        target_modality,
                        category_num,
                        start_ind,
                        end_ind,
                        output_dir='./',
                        ):

    """
    Note
    ----
    `mean_all` is average of latent embedding corresponding to the label.
    The indices of `mean_all` are correspond to the number label.
    """
    def check_satisfied_equation(base, add, sub):
        over_one = 1 <= base + add - sub
        under_nine = base + add - sub <= 9
        no_repeat_base = base != sub
        return over_one and under_nine and no_repeat_base

    mean_all = [None for i in range(category_num)]
    rslt = {}
    generations = []
    target_calcs = [
        # base, add, sub
        [1, 1, 1],
        [1, 9, 8],
        [1, 8, 4],
        [2, 7, 1],
        [2, 7, 2],
        [3, 3, 3],
        [3, 5, 2],
        [3, 6, 3],
        [5, 3, 5],
        [9, 7, 8],
        # [0, 3, 2],  # error
    ]

    l = range(1, 10)

    # make combinations of calculation equation
    elements = [(base, add, sub) \
                for base, add, sub in itertools.product(l, l, l) \
                if check_satisfied_equation(base, add, sub)]
    labels = [base + add - sub for base, add, sub in elements]
    print('length of elements:', len(elements), len(labels))
    for number in l:
        print('length of ' + str(number) + ':', str(labels.count(number)))
    print('all elements:\n')
    pprint.pprint(list(zip(labels, elements)))

    # latent embedding for each number
    for i in range(start_ind, end_ind):
        print('i:', str(i))
        target_latents = latent_all[np.where(label_all == i)]
        mean_all[i] = np.mean(target_latents, axis = 0)
        # print(np.mean(target_latents, axis = 1))
        print('shape', mean_all[i].shape)
    print('mean_all:', mean_all)

    # analyse for selected calculation
    for target_calc in target_calcs:
        print('---')
        rslt.update(check_selected_calculation(
            runner=runner,
            classifier=classifier,
            mean_all=mean_all,
            base=target_calc[0],
            add=target_calc[1],
            sub=target_calc[2],
            target_modality=target_modality,
            output_dir=output_dir,
        ))

    # latent embedding for each calculation
    print('---')
    for base, add, sub in elements:
        _, generation, _ = calculate_simple(
            runner=runner,
            mean_all=mean_all,
            base=base,
            add=add,
            sub=sub,
            target_modality=target_modality,
            num_data=1,
            output_dir=None,
        )
        generations.append(generation)
    # print(torch.stack(generations).shape)

    # analyse accuracy
    acc, confusion = count_reconst(
        recon=torch.stack(generations).to(runner.args.device)[:,0,:,:],
        true_label=torch.tensor(labels).to(runner.args.device),
        classifier=classifier,
        output_dir=output_dir,
        suffix='_mathe',
    )
    rslt.update({
        'mathematics_avg': acc,
        'mathematics_all': np.nan,
      })
    return rslt

def calculate_simple(runner,
                     mean_all,
                     base,
                     add,
                     sub,
                     num_data=64,
                     target_modality=1,
                     output_dir=None,
                     ):
    true_label = base + add - sub
    mean = mean_all[base] + mean_all[add] - mean_all[sub]
    suffix = str(base) + '+' + str(add) + '-' + str(sub)
    if 'MMVAE' in runner.model_name:
        generation = runner.model.generate_special(
            mean=mean,
            num_data=num_data,
            target_modality=target_modality,
            output_dir=output_dir,
            suffix=suffix,
        )
        generation = generation[target_modality]
    else:
        generation = runner.model.generate_special(
            mean=mean,
            num_data=num_data,
            output_dir=output_dir,
            suffix=suffix,
        )
    return mean, generation, true_label

def check_selected_calculation(runner,
                   classifier,
                   mean_all,
                   base,
                   add,
                   sub,
                   target_modality=None,
                   num_data=64,
                   output_dir='./',
                   ):
    suffix = str(base) + '+' + str(add) + '-' + str(sub)
    mean = mean_all[base] + mean_all[add] - mean_all[sub]
    labels = [base + add - sub for idx in range(num_data)]
    print('Target calculation:', suffix)

    # mean_all_for_calc = np.array(mean_all[start_ind:])
    # model.generate_special(mean_all[0])
    # base = mean_all_for_calc.mean(axis = 0)
    # base[8] = (mean_all[1] + mean_all[8]  - mean_all[4])[8]
    # base[8] = (mean_all[0] + mean_all[7] - mean_all[4])[8]
    # model.generate_special(base)

    _, generation, _ = calculate_simple(
        runner=runner,
        mean_all=mean_all,
        base=base,
        add=add,
        sub=sub,
        target_modality=target_modality,
        num_data=64,
        output_dir=output_dir,
    )
    acc, confusion = count_reconst(
        recon=generation.to(runner.args.device),
        true_label=torch.tensor(labels).to(runner.args.device),
        classifier=classifier,
        output_dir=output_dir,
        suffix='_' + suffix,
        set_label=list(range(1, 10)),
    )
    """
    points = TSNE(n_components=3, random_state=0).fit_transform(mean_all)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0], points[:,1], points[:, 2], s = 0.5, c = color_dict[i])
    """
    start_ind, end_ind = 1, 10
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
    plt.clf()

    check_distance(
        mean_all=mean_all,
        base=base,
        add=add,
        sub=sub,
    )

    return {
        'mathematics_' + suffix + '_avg': acc,
        'mathematics_' + suffix + '_all': np.nan,
    }

def check_distance(mean_all,
                   base,
                   add,
                   sub,
                   ):
    """Perform calculation of `base + add - sub` and check distance between ideal (target) and calculated (result) latent embedding.
    """
    result = mean_all[base] + mean_all[add] - mean_all[sub]
    target = mean_all[base + add - sub]

    min_dist, min_ind = 1e9, -1
    # for i in range(9):  # original code by Noda-san
    for i in range(1, 10):
        # print(type(mean_all[i]))
        dist = np.linalg.norm(mean_all[i] - result)
        # print(i, dist)
        if dist < min_dist:
            min_ind = i
            min_dist = dist
    print(
        'Minimum distance of calculation.',
        '\ntrue answer:', base + add - sub,
        '\nindices:', min_ind,
        '\ndistance:', min_dist,
    )
    return


def get_latent_space(runner,
                     target_modality,
                     target_property,
                     category_num):

    label_all = []
    latent_all = []

    # Settings for label and latent states
    for i, dataT in enumerate(runner.train_loader):
        data, label = unpack_data(
            dataT,
            device=runner.args.device,
            require_label=True,
            target=target_modality
        )

        # check data structure and content
        if i == 0:
            print(
                '\n===\nData and label info...',
                '\ntype(label):', type(label),
                '\nlabel:', label,
                # '\nlabel[0]:', label[0],
                # '\nlabel[1]:', label[1],
                '\ntype(data):', type(data),
                # '\ndata[0]:', data[0],
                '\ndata[0].shape:', data[0].shape
            )

        label = convert_label_to_int(
            label=label,
            model_name=runner.model_name,
            target_property=target_property,
            target_modality=target_modality,
            data=data,
        )

        # Get latent space
        # runner.model.reconstruct(data, '.', 3939, n =10)
        # pil_image.save(output_dir + '/check_' + str(label[indr])+ '.png')
        latent_space = runner.model.latent(data)[target_modality].cpu().detach().numpy()
        latent_dim  = latent_space.shape[-1]

        if runner.model_name == 'mnist_clevr':
            label = label[0]
        try:
            label_all += list(np.array(label.cpu()))
        except:
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
        runner,
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
    if runner.model_name == 'clevr':
        start_ind, end_ind = 3, 11
    elif runner.model_name == 'mnist_clevr':
        start_ind, end_ind = 3, 10

    return (start_ind, end_ind)


def analyse_model(runner,
                  classifier,
                  target_modality,
                  target_property,
                  output_dir='./'):

    # What analysis are performed?
    require_reconst = True if target_property == 1 else False
    require_cross = True \
        if runner.model_name == 'MMVAE_CMNIST_OSCN' and \
           target_property == 1 else False
    require_classifier = False
    # require_classifier = False \ # TODO
    #     if runner.model_name == 'MMVAE_CMNIST_OSCN' else False
    require_magnitude = True
    require_2d = True
    require_3d = False  # if true, error happened.
    require_cluster = True
    require_mathematics = True if target_property == 1 else False
    withzero = False

    # Settings for the number of cateogry
    if target_property == 0:
        category_num = 4
    elif target_property == 1:
        category_num = 10
    elif target_property == 2:
        category_num = 3
        if runner.model_name == 'clevr' or runner.model_name == 'mnist_clevr':
            category_num = 8
    else:
        raise Exception

    # Start and end index of category number
    start_ind, end_ind =  get_index(
        target_modality=target_modality,
        target_property=target_property,
        runner=runner,
        withzero=withzero,
    )

    # Get latent space
    latent_all, label_all = get_latent_space(
        runner=runner,
        target_modality=target_modality,
        target_property=target_property,
        category_num=category_num,
    )

    # np.save('latent_' + runner.model_name + "_" + str(target_modality), latent_all)
    # np.save('label_' + runner.model_name + "_" + str(target_modality), np.array(label_all))

    print(
        '\n===\nSetting info...',
        '\nTarget modality:', target_modality,
        '\nTarget property:', target_property,
        '\nOutput directory:', output_dir,
        '\nNumber of category', category_num,
        '\nShape of latent_all:', latent_all.shape,
        '\nShape of label_all: ', label_all.shape,
        '\nSet of label_all: ', set(label_all),
        '\nStart index:', start_ind,
        '\nEnd index:', end_ind,
        '\nwithzero:', withzero
    )

    # Main analysis
    rslt = {}
    if require_reconst:
        print('---')
        rslt.update(analyse_reconst(
            runner=runner,
            classifier=classifier,
            target_modality=target_modality,
            output_dir=output_dir,
        ))
    if require_cross:
        print('---')
        rslt.update(analyse_cross(
            runner=runner,
            classifier=classifier,  # classifier for target_modality_to
            target_modality_from=1 - target_modality,
            target_modality_to=target_modality,
            output_dir=output_dir,
        ))
    if require_classifier:
        print('---')
        rslt.update(analyse_classifier(
            classifier=classifier,
            output_dir=output_dir,
        ))
    if require_cluster:
        print('---')
        rslt.update(analyse_cluster(
            latent_all=latent_all,
            label_all=label_all,
            target_modality=target_modality,
            target_property=target_property,
            output_dir=output_dir,
        ))
    if require_magnitude:
        print('---')
        rslt.update(analyse_magnitude(
            runner=runner,
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_property=target_property,
            start_ind=start_ind,
            end_ind=end_ind,
            withzero=withzero,
            output_dir=output_dir,
        ))
    if require_2d:
        print('---')
        rslt.update(analyse_tsne_2d(
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_modality=target_modality,
            target_property=target_property,
            start_ind=start_ind,
            end_ind=end_ind,
            output_dir=output_dir,
        ))
    if require_3d:
        print('---')
        rslt.update(analyse_tsne_3d(
            label_all=label_all,
            latent_all=latent_all,
            category_num=category_num,
            target_modality=target_modality,
            target_property=target_property,
            output_dir=output_dir,
        ))
    if require_mathematics:
        print('---')
        rslt.update(analyse_mathematics(
            runner=runner,
            classifier=classifier,
            latent_all=latent_all,
            label_all=label_all,
            target_modality=target_modality,
            category_num=category_num,
            start_ind=start_ind,
            end_ind=end_ind,
            output_dir=output_dir,
        ))
    # save
    avgs = {
        'id': runner.args.run_id,
        'model_name': runner.model_name,
        'target_modality': target_modality,
        'target_property': target_property,
    }
    avgs.update(dict(filter(lambda item: 'avg' in item[0], rslt.items())))
    print('results (all):', rslt)
    print('results (only averages):', avgs)
    with open(output_dir + '/analyse_result.csv', 'w') as f:
        w = csv.DictWriter(f, avgs.keys())
        w.writeheader()
        w.writerow(avgs)
    return rslt

def analyse(args,
            args_classifier_cmnist,
            args_classifier_oscn,
            ):
    """Main function for model analysis
    target_modality
        Whether latent states are analysed
        Only Single modal (OSCN or CMNIST) -> 0
        CMNIST_OSCN -> cmnist (0) or oscn(1)?

    target_property
        Whether information are analysed, color(0), number(1), shape(2)?
    """
    runner = Runner(args=args)

    if runner.model_name == 'MMVAE_CMNIST_OSCN':
        modality_list = [0, 1]
    elif runner.model_name == 'VAE_CMNIST' or runner.model_name == 'VAE_OSCN':
        modality_list = [0]
    else:
        Exception

    for target_modality in modality_list:
        if (runner.model_name == 'MMVAE_CMNIST_OSCN' and target_modality == 0) or \
           runner.model_name == 'VAE_CMNIST':
            classifier = Runner(args=args_classifier_cmnist)
        elif (runner.model_name == 'MMVAE_CMNIST_OSCN' and target_modality == 1) or \
             runner.model_name == 'VAE_OSCN':
            classifier = Runner(args=args_classifier_oscn)
        else:
            Exception

        if runner.model_name == 'MMVAE_CMNIST_OSCN' or \
           runner.model_name == 'VAE_OSCN':
            property_list = [0, 1, 2]
        elif runner.model_name == 'VAE_CMNIST':
            property_list = [0, 1]
        else:
            Exception

        for target_property in property_list:
            output_dir = args.output_dir + '/' + \
                str(target_modality) + '_' + str(target_property)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            sys.stdout = Logger('{}/analyse.log'.format(output_dir))
            print(
                '\n\n',
                '==============\n',
                'START ANALYSIS\n',
                '==============\n',
            )
            rslt = analyse_model(
                runner=runner,
                classifier=classifier,
                target_modality=target_modality,
                target_property=target_property,
                output_dir=output_dir,
            )
    return


if __name__ == '__main__':
    args = {}
    visualize_latent(args)
