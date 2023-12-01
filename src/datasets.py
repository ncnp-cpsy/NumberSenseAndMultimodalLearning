import io
import json
import os
import pickle
from collections import Counter, OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets


class DatasetOSCN(Dataset):
    def __init__(self,
                 train):
        if train:
            self.images = torch.load('./data/oscn_train_images.pt')
            self.labels = np.load('./data/oscn_train_labels.npy',
                                  allow_pickle=True)
        else:
            self.images = torch.load('./data/oscn_test_images.pt')
            self.labels = np.load('./data/oscn_test_labels.npy',
                                  allow_pickle=True)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class DatasetCMNIST(Dataset):
    def __init__(self, train):
        if train:
            self.images = torch.load('./data/cmnist_train_images.pt')
            self.labels = torch.load('./data/cmnist_train_labels.pt')
        else:
            self.images = torch.load('./data/cmnist_test_images.pt')
            self.labels = torch.load('./data/cmnist_test_labels.pt')

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSentences(Dataset):

    def __init__(self, root_data_dir, split, transform=None, **kwargs):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = os.path.join(root_data_dir, 'cub')
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 32)
        self.min_occ = kwargs.get('min_occ', 3)
        self.transform = transform
        os.makedirs(os.path.join(root_data_dir, "lang_emb"), exist_ok=True)

        self.gen_dir = os.path.join(self.data_dir, "oc:{}_msl:{}".
                                    format(self.min_occ, self.max_sequence_length))

        if split == 'train':
            self.raw_data_path = os.path.join(self.data_dir, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.raw_data_path = os.path.join(self.data_dir, 'text_testclasses.txt')
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'cub.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[str(idx)]['length']

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()


class CUBImageFt(Dataset):
    def __init__(self, root_data_dir, split, device):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = os.path.join(root_data_dir, 'cub')
        self.data_file = os.path.join(self.data_dir, split)
        self.gen_dir = os.path.join(self.data_dir, 'resnet101_2048')
        self.gen_ft_file = os.path.join(self.gen_dir, '{}.ft'.format(split))
        self.gen_data_file = os.path.join(self.gen_dir, '{}.data'.format(split))
        self.split = split

        tx = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(self.data_file, transform=tx)

        os.makedirs(self.gen_dir, exist_ok=True)
        if not os.path.exists(self.gen_ft_file):
            print("Data file not found for CUB image features at `{}`. "
                  "Extracting resnet101 features from CUB image dataset... "
                  "(this may take a while)".format(self.gen_ft_file))
            self._create_ft_mat(device)

        else:
            self._load_ft_mat()

    def __len__(self):
        return len(self.ft_mat)

    def __getitem__(self, idx):
        return self.ft_mat[idx]

    def _load_ft_mat(self):
        self.ft_mat = torch.load(self.gen_ft_file)

    def _load_data(self):
        self.data_mat = torch.load(self.gen_data_file)

    def _create_ft_mat(self, device):
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.model.eval()

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=256,
                                             shuffle=False, **kwargs)
        with torch.no_grad():
            ft_mat = torch.cat([self.model(data[0]).squeeze() for data in loader])

        torch.save(ft_mat, self.gen_ft_file)
        del ft_mat

        data_mat = torch.cat([data[0].squeeze() for data in loader])
        torch.save(data_mat, self.gen_data_file)

        self._load_ft_mat()


def convert_label_to_int(label, model_name, target_property, data=None):
    do_print = False
    zukei_to_int = {
        'j': 0,
        's': 1,
        't': 2,
    }
    color_to_int = {
        'r': 3,
        'g': 2,
        'b': 1,
        'w': 0,
    }

    if do_print:
        print(
            '\n===\nlabel info before conversion...',
            '\nmodel name:', model_name,
            '\ntarget_property:', target_property,
            '\ntype(label):', type(label),
            '\nlabel:', label,
            '\nlabel[0]:', label[0],
            '\nlabel[1]:', label[1],
        )

    if model_name == 'MMVAE_CMNIST_OSCN':
        if target_property == 1:
            print('AAAAAAAAAAAAA')
            label = label[0]
        else:
            label = label[1]  # 全部の情報が必要

    # Pattern of label -> [0,1,3,,,] (CMNIST and CLEVR)
    if (model_name == 'Classifier_CMNIST') or \
       (model_name == 'VAE_CMNIST') or \
       (model_name == 'VAE_CLEVR') or \
       (model_name == 'MMVAE_CMNIST_OSCN' and target_modality == 0) or \
       (model_name == 'MMVAE_MNIST_CLEVR'):
        # (model_name == 'MMVAE_CMNIST_OSCN' and target_property == 1) or \
        if target_property == 0:
            label = []
            if model_name == 'MMVAE_CMNIST_OSCN' :
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
            if (model_name == 'Classifier_CMNIST') or \
               (model_name == 'VAE_CMNIST'):
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
    elif (model_name == 'Classifier_OSCN') or \
         (model_name == 'VAE_OSCN') or \
         (model_name == 'MMVAE_CMNIST_OSCN' and target_modelity == 1):
         # (model_name == 'MMVAE_CMNIST_OSCN' and target_property != 1):
        label = list(label)
        if target_property == 0:
            label = [color_to_int[s[0]] for s in label]
        elif target_property == 1:
            label = [int(s[1]) for s in label]
        elif target_property == 2:
            label = [zukei_to_int[s[2]] for s in label]
        else:
            raise Exception
    else:
        raise Exception

    if type(label) == list:
        label = torch.Tensor(label)

    if do_print:
        print(
            '\n===\nlabel info after conversion...',
            '\ntype(label):', type(label),
            '\nlabel:', label,
            '\nlabel[0]:', label[0],
            '\nlabel[1]:', label[1],
        )

    return label
