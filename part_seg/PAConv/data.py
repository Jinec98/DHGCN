import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from part_utils import *
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetPart(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False, split_num=5):
        self.npoints = npoints
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize
        self.split_num = split_num
        self.max_distance = self.split_num + 1 

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # split part
        load_partdata = True
        if load_partdata and os.path.exists('data/p2v_indices_{}_splitnum_{}_md{}.npy'.format(split, self.split_num, self.max_distance)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/p2v_indices_{}_splitnum_{}_md{}.npy'.format(split, self.split_num, self.max_distance), allow_pickle=True)
            self.part_distance = np.load(
                'data/part_distance_{}_splitnum_{}_md{}.npy'.format(split, self.split_num, self.max_distance), allow_pickle=True)
        else:
            self.datapath = np.array(self.datapath, dtype=object)
            data_all = [np.loadtxt(data_root) for data_root in self.datapath[:, 1]]
            
            self.p2v_indices = np.empty(len(data_all), object)
            self.part_distance = np.empty((len(data_all), self.split_num**3, self.split_num**3))
            print('Split point cloud to part... (one by one)')
            for i, data in enumerate(tqdm(data_all)):
                data = np.array(data).astype(np.float32)[np.newaxis, :, 0:3]
                p2v_indices_i, part_distance_i = split_part(data, self.split_num, self.max_distance)
                self.p2v_indices[i] = p2v_indices_i
                self.part_distance[i] = part_distance_i
            np.save('data/p2v_indices_{}_splitnum_{}_md{}.npy'.format(split, self.split_num, self.max_distance),
                        self.p2v_indices)
            np.save('data/part_distance_{}_splitnum_{}_md{}.npy'.format(split, self.split_num, self.max_distance),
                    self.part_distance)
            print('Split part done!')
        
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        part_distance = self.part_distance[index].squeeze()  # (1, 27, 27)
        p2v_indices = self.p2v_indices[index].squeeze()  # (1, N)
        
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)
        
        # sample to num_points
        if self.npoints <= point_set.shape[0]:
            sample_ids = np.random.permutation(point_set.shape[0])[:self.npoints]
        else:
            sample_ids = np.random.choice(point_set.shape[0], self.npoints, replace=True)
       
        #resample
        point_set = point_set[sample_ids, :]
        seg = seg[sample_ids]
        normal = normal[sample_ids, :]
        p2v_indices = p2v_indices[sample_ids]
        
        
        part_rand_idx = np.arange(self.split_num**3)
        np.random.shuffle(part_rand_idx)
        part_distance = part_distance[part_rand_idx][:, part_rand_idx]

        return cat, point_set, cls, seg, normal, p2v_indices, part_distance, part_rand_idx

    def __len__(self):
        return len(self.datapath)
