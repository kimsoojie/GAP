import numpy as np

from torch.utils.data import Dataset
from utils.utils import get_label_split_oneshot, get_label_split_zsl
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, true_only=False):
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
        :param true_only: If true, load only samples whose labels are provided true labels.
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
        self.true_only = true_only
        self.unseen_split = 100
        #self.seen_labels, self.unseen_labels = get_label_split_zsl(data_path, self.unseen_split)
        #self.seen_labels, self.unseen_labels = get_label_split_oneshot(data_path, self.unseen_split)
        
        self.load_data()
        #self.load_data_split_zsl()
        #self.load_data_split_oneshot(split=self.unseen_split)
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Output format: N, C, T, V, M. CARE-PD uses V=22 and M=1.
        npz_data = np.load(self.data_path, allow_pickle=True)
        if 'x_train' in npz_data and 'x_test' in npz_data:
            if self.split == 'train':
                self.data = npz_data['x_train']
                self.label = np.where(npz_data['y_train'] > 0)[1]
                self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            elif self.split == 'test':
                self.data = npz_data['x_test']
                self.label = np.where(npz_data['y_test'] > 0)[1]
                self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            else:
                raise NotImplementedError('data split only supports train/test')
            self.data = self._format_skeleton_array(self.data)
        elif 'skeletons' in npz_data and 'labels' in npz_data:
            all_data = npz_data['skeletons']
            all_labels = npz_data['labels'].astype(np.int64)

            indices = np.arange(len(all_data))

            if self.true_only:
                if 'true_mask' in npz_data:
                    true_mask = npz_data['true_mask'].astype(bool)
                elif 'label_sources' in npz_data:
                    true_mask = np.asarray(npz_data['label_sources']).astype(str) == 'true'
                else:
                    raise KeyError('true_only=True requires true_mask or label_sources in CARE-PD npz')
                indices = indices[true_mask[indices]]

            # Deterministic stratified 9:1 train/test split.
            split_seed = 42
            test_ratio = 0.1
            rng = np.random.default_rng(split_seed)

            train_parts = []
            test_parts = []

            subset_labels = all_labels[indices]
            for label in np.unique(subset_labels):
                label_indices = indices[subset_labels == label].copy()
                rng.shuffle(label_indices)

                n_test = max(1, int(round(len(label_indices) * test_ratio))) if len(label_indices) > 1 else 0
                test_parts.append(label_indices[:n_test])
                train_parts.append(label_indices[n_test:])

            train_indices = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
            test_indices = np.concatenate(test_parts) if test_parts else np.empty((0,), dtype=np.int64)

            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

            if self.split == 'train':
                indices = train_indices
            elif self.split == 'test':
                indices = test_indices
            else:
                raise NotImplementedError('data split only supports train/test')

            self.data = self._format_skeleton_array(all_data[indices])
            self.label = all_labels[indices]

            sample_ids = npz_data['sample_ids'][indices] if 'sample_ids' in npz_data else indices
            self.sample_name = [str(x) for x in sample_ids]

    @staticmethod
    def _format_skeleton_array(data):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 5:
            return data
        if data.ndim == 4:
            # N, T, V, C -> N, C, T, V, 1
            if data.shape[-1] in (2, 3):
                return data.transpose(0, 3, 1, 2)[:, :, :, :, None]
            raise ValueError(f'Unsupported 4D CARE-PD skeleton shape: {data.shape}')
        if data.ndim == 3:
            N, T, D = data.shape
            if D % (22 * 3) == 0:
                M = D // (22 * 3)
                return data.reshape((N, T, M, 22, 3)).transpose(0, 4, 1, 3, 2)
            if D % (25 * 3) == 0:
                M = D // (25 * 3)
                return data.reshape((N, T, M, 25, 3)).transpose(0, 4, 1, 3, 2)
        raise ValueError(f'Unsupported CARE-PD skeleton shape: {data.shape}')

    def load_data_split_zsl(self, split=10):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        # seen/unseen split
        # split 115/5: unseen [0, 24, 48, 72, 96]
        # split 110/10: unseen [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
        # split 96/24: unseen [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
        # split 80/40: unseen [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117]
        # split 60/60: unseen [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118]

        # unique_labels = np.unique(self.label)
        # if split == 5: num_unseen=5
        # elif split == 10: num_unseen=10
        # elif split == 24: num_unseen=24
        # elif split == 40: num_unseen=40
        # elif split == 60: num_unseen=60
        
        # step = max(1, len(unique_labels) // num_unseen)
        # unseen_labels = set(unique_labels[::step][:num_unseen]) 
        # seen_labels = set(unique_labels) - unseen_labels
        # print(f"Unseen labels: {sorted(unseen_labels)}")
        # print(f"Seen labels: {sorted(seen_labels)}")
        
        seen_labels = self.seen_labels
        unseen_labels = self.unseen_labels

        if self.split == 'train': # seen data in train set
            indices = [i for i, label in enumerate(self.label) if label in seen_labels]
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.sample_name = [self.sample_name[i] for i in indices]

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    
    def load_data_split_oneshot(self, split=20):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        # unique_labels = list(range(120))
        # base_unseen_labels = list(range(0, 120, 6)) 
        
        # if split == 20: num_seen=20
        # elif split == 40: num_seen=40
        # elif split == 60: num_seen=60
        # elif split == 80: num_seen=80
        # elif split == 100: num_seen=100
        
        # remaining_labels = [l for l in unique_labels if l not in base_unseen_labels]
        # if len(remaining_labels) >= num_seen:
        #     indices = np.linspace(0, len(remaining_labels)-1, num_seen, dtype=int)
        #     seen_labels = [remaining_labels[i] for i in indices]
        # else:
        #     seen_labels = remaining_labels
        # unseen_labels = set(base_unseen_labels)
        # print(f"Unseen labels: {sorted(unseen_labels)}")
        # print(f"Seen labels: {sorted(seen_labels)}")
        
        seen_labels = self.seen_labels
        unseen_labels = self.unseen_labels

        if self.split == 'train': # seen data in train set
            indices = [i for i, label in enumerate(self.label) if label in seen_labels]
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.sample_name = [self.sample_name[i] for i in indices]

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
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
        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # # reshape Tx(MVC) to CTVM
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # if self.random_rot:
        #     data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import care_pd_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in care_pd_pairs:
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



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
