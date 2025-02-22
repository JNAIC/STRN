import os.path

import numpy as np
import torch

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, jointbone=False):
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
        self.jointbone = jointbone
        self.load_data()
        self.graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))
        # self.load_per_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.split == 'train':
            # 加载数据
            self.data = np.load(os.path.join(self.data_path, 'train_joint.npy'), mmap_mode='r')
            self.label = np.load(os.path.join(self.data_path, 'train_label.npy'), mmap_mode='r')
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]

        elif self.split == 'test':
            self.data = np.load(os.path.join(self.data_path, 'test_joint.npy'), mmap_mode='r')
            self.label = np.load(os.path.join(self.data_path, 'test_label.npy') ,mmap_mode='r')
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

    def load_per_data(self):
        if self.split == 'train':
            self.data_dir = os.path.join(self.data_path, 'train')
        elif self.split == 'test':
            self.data_dir = os.path.join(self.data_path, 'test')
        else:
            raise NotImplementedError('data split only supports train/test')
        self.length = len(os.listdir(self.data_dir))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)
        # return self.length

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data = np.load(os.path.join(self.data_dir, f'{self.split}_{index}.npz'))
        # data_numpy = data['x']
        # T, _ = data_numpy.shape
        # data_numpy = data_numpy.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)
        # label = data['y']

        data_numpy = self.data[index]
        label = self.label[index]

        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        joint = data_numpy
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            joint = data_numpy
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.graph:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        
        if self.jointbone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.graph:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = np.concatenate((bone_data_numpy, data_numpy), axis=0)

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


if __name__ == '__main__':
    feeder = Feeder(data_path='../data', split= 'train', debug=False, random_choose=False, random_shift=False,
                    random_move=False, window_size=64, normalization=False, random_rot=True, p_interval=[0.5, 1])
    print(feeder[13619])