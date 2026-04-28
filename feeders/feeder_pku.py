import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from utils.utils import get_label_split_oneshot, get_label_split_zsl

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, unseen_split=0, split_method='zsl'):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.unseen_split = unseen_split
        self.split_method = split_method
        
        if split == 'train':
            if self.split_method == 'zsl':
                self.seen_labels, self.unseen_labels = get_label_split_zsl(data_path, self.unseen_split)
            elif self.split_method == 'oneshot':
                self.seen_labels, self.unseen_labels = get_label_split_oneshot(data_path, self.unseen_split)
            
            if self.split_method == 'zsl':
                self.load_data_split_zsl()
            elif self.split_method == 'oneshot':
                self.load_data_split_oneshot()
            elif self.split_method == 'all':
                self.load_data()
        elif split == 'test':
            self.load_data()
        
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train'].squeeze(1)
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = npz_data['y_test'].squeeze(1)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        # self._subsample_per_class(20)
        
        # N, T, _ = self.data.shape
        # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def load_data_split_zsl(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train'].squeeze(1)
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = npz_data['y_test'].squeeze(1)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        seen_labels = self.seen_labels
        unseen_labels = self.unseen_labels

        if self.split == 'train': # seen data in train set
            indices = [i for i, label in enumerate(self.label) if label in seen_labels]
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.sample_name = [self.sample_name[i] for i in indices]

        # N, T, _ = self.data.shape
        # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    
 
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def _subsample_per_class(self, max_per_class=100):
       
        if max_per_class is None or max_per_class <= 0:
            return
        labels = np.array(self.label)
        unique = np.unique(labels)
        keep_indices = []
        for lab in unique:
            idx = np.where(labels == lab)[0]
            n = len(idx)
            if n > max_per_class:
                chosen = np.linspace(0, n - 1, max_per_class, dtype=int)
                selected = idx[chosen]
            else:
                selected = idx
            keep_indices.append(selected)
        if len(keep_indices) == 0:
            return
        keep = np.concatenate(keep_indices)
        keep.sort()
        self.data = self.data[keep]
        self.label = self.label[keep]
        if hasattr(self, 'sample_name'):
            self.sample_name = [self.sample_name[i] for i in keep]

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
