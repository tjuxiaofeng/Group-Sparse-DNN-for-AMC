import torch
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import pickle
import sklearn.preprocessing


class RML2016b(data.Dataset):
    def __init__(self, transforms=None, train=True, fine_train=False, conf_class=[]):
        xd = pickle.load(open("data/RML2016.10b.dat", 'rb'), encoding='iso-8859-1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0])
        if fine_train:
            mods = conf_class
        self.classes = mods
        x = []
        lbl = []
        for mod in mods:
            for snr in snrs[10:19]:
                x.append(xd[(mod, snr)])
                for i in range(xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        x = np.vstack(x)
        x = x.reshape((-1, 256))
        scaler = sklearn.preprocessing.MinMaxScaler()
        x = scaler.fit_transform(x)
        x = x.reshape((-1, 1, 2, 128))

        np.random.seed(2016)
        n_examples = x.shape[0]
        n_train = n_examples * 0.7
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        x_train = x[train_idx]
        x_test = x[test_idx]

        y_train = list(map(lambda ii: mods.index(lbl[ii][0]), train_idx))
        y_test = list(map(lambda ii: mods.index(lbl[ii][0]), test_idx))

        if train:
            self.data = x_train
            self.label = y_train
        else:
            self.data = x_test
            self.label = y_test


    def __getitem__(self, item):
        data = self.data[item]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.data)

    def __getmods__(self):
        mods = self.classes
        return mods
