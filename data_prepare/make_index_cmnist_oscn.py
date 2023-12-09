import torch
from torchvision import datasets, transforms
import numpy as np

def rand_match_on_idx(l1, idx1, l2, idx2, max_d, dm, cs):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    #cs = ['w']
    _idx1, _idx2 = [], []

    for l in l1.unique():  # assuming both have same idxs
        for c in range(len(cs)):
            l_idx1 = idx1[(l1 == l) & (c * int(len(idx1) / len(cs)) <= idx1) & ( idx1 <  (c+1) * int(len(idx1) / len(cs)) ) ] # 数字がlで色がcのが抽出される
            tar_class =  cs[c] + str(l.item())

            l_idx2 = []
            for m in range(len(l2)):
                if l2[m][:2] == tar_class:
                    l_idx2.append(idx2[m])

            l_idx2 = torch.Tensor(l_idx2)

            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

if __name__ == '__main__':
    max_d = 10000  # maximum number of datapoints per class
    dm = 2        # data multiplier: random permutations to match
    cs = ['r','g','b', 'w']
    #cs = ['w']

    train_cmnist_labels = torch.load('../data/cmnist_train_labels.pt')
    train_oscn_labels = np.load('../data/oscn_train_labels.npy',allow_pickle=True)

    test_cmnist_labels = torch.load('../data/cmnist_test_labels.pt')
    test_oscn_labels = np.load('../data/oscn_test_labels.npy', allow_pickle=True)


    cmnist_l, cmnist_li = train_cmnist_labels.sort()
    oscn_l, oscn_li = np.sort(train_oscn_labels),np.argsort(train_oscn_labels)
    print(len(cmnist_l), len(cmnist_li))
    print(len(oscn_l), len(oscn_li) )
    idx1, idx2 = rand_match_on_idx(cmnist_l, cmnist_li, oscn_l, oscn_li, max_d=max_d, dm=dm, cs= cs)
    print('len train idx:', len(idx1), len(idx2))
    print('testします')

    for i in [np.random.randint(0,len(idx1)) for i in range(5)]:
        print(cs[int(idx1[i]/(len(train_cmnist_labels)/len(cs)))] + str(train_cmnist_labels[idx1[i]].item())  )
        print(train_oscn_labels[int(idx2[i].item())][:2] )
    torch.save(idx1.int(), '../data/train-ms-cmnist-idx.pt')
    torch.save(idx2.int(), '../data/train-ms-oscn-idx.pt')

    cmnist_l, cmnist_li = test_cmnist_labels.sort()
    oscn_l, oscn_li = np.sort(test_oscn_labels),np.argsort(test_oscn_labels)
    idx1, idx2 = rand_match_on_idx(cmnist_l, cmnist_li, oscn_l, oscn_li, max_d=max_d, dm=dm, cs= cs)
    print('len test idx:', len(idx1), len(idx2))
    torch.save(idx1.int(), '../data/test-ms-cmnist-idx.pt')
    torch.save(idx2.int(), '../data/test-ms-oscn-idx.pt')
