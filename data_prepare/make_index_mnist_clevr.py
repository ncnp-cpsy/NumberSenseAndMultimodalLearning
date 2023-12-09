import torch
from torchvision import datasets, transforms
import numpy as np

def rand_match_on_idx(l1, idx1, l2, idx2, max_d, dm):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []

    for l in np.unique(l1):  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

if __name__ == '__main__':
    max_d = 10000  # maximum number of datapoints per class
    dm = 2        # data multiplier: random permutations to match


    train_mnist = datasets.MNIST('../data', train=True, download=True)
    test_mnist = datasets.MNIST('../data', train=False, download=True)

    train_mnist_labels = train_mnist.targets
    train_clevr_labels = torch.load('../data/clevr_train_labels.pt')
    
    test_mnist_labels = test_mnist.targets
    test_clevr_labels = torch.load('../data/clevr_test_labels.pt')


    mnist_l, mnist_li = train_mnist_labels.sort()
    clevr_l, clevr_li = np.sort(train_clevr_labels),np.argsort(train_clevr_labels)
    print(len(mnist_l), len(mnist_li))
    print(len(clevr_l), len(clevr_li) )
    idx1, idx2 = rand_match_on_idx(clevr_l, clevr_li, mnist_l, mnist_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2))
    print('testします')

    for i in [np.random.randint(0,len(idx1)) for i in range(5)]:
        print(train_clevr_labels[int(idx1[i].item())] )
        print(train_mnist_labels[int(idx2[i].item())])
    
    torch.save(idx1.int(), '../data/train-ms-clevr-idx.pt')
    torch.save(idx2.int(), '../data/train-ms-mnist-idx.pt')

    mnist_l, mnist_li = test_mnist_labels.sort()
    clevr_l, clevr_li = np.sort(test_clevr_labels),np.argsort(test_clevr_labels)
    idx1, idx2 = rand_match_on_idx(clevr_l, clevr_li, mnist_l, mnist_li, max_d=max_d, dm=dm)
    print('len test idx:', len(idx1), len(idx2))
    
    torch.save(idx1.int(), '../data/test-ms-clevr-idx.pt') 
    torch.save(idx2.int(), '../data/test-ms-mnist-idx.pt')