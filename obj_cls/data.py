import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from util import *
from part_utils import *


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))
  
      
def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'train':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, split_num=3, partition='train'):
        self.data, self.label = load_data_cls(partition)  # data: (B,N,C) numpy

        self.num_points = num_points
        self.partition = partition
        self.data = self.data[:, :self.num_points]
        self.split_num = split_num
        self.max_distance = self.split_num + 1  # 0,1,2,3

        load_partdata = True
        if load_partdata and os.path.exists('data/ModelNet40_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/ModelNet40_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance))
            self.part_distance = np.load(
                'data/ModelNet40_part_distance_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance))
        else:
            self.p2v_indices, self.part_distance = split_part(self.data, self.split_num, self.max_distance)
            if not self.debug:
                np.save('data/ModelNet40_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance),
                        self.p2v_indices)
                np.save('data/ModelNet40_part_distance_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance),
                        self.part_distance)
            print('Split part done!')
            

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        part_distance = self.part_distance[item]  # (1, 27, 27)
        p2v_indices = self.p2v_indices[item]  # (1, N)


        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            # shuffle pointcloud along N dim
            pc_rand_idx = np.arange(self.num_points)
            np.random.shuffle(pc_rand_idx)
            pointcloud = pointcloud[pc_rand_idx, :]
            p2v_indices = p2v_indices[pc_rand_idx]

            # shuffle parts along N dim
            part_rand_idx = np.arange(self.split_num**3)          
            np.random.shuffle(part_rand_idx)
            part_distance = part_distance[part_rand_idx][:, part_rand_idx]
        else:
            part_rand_idx = np.arange(self.split_num**3)

        return (pointcloud, label, p2v_indices, part_distance, part_rand_idx)

    def __len__(self):
        return self.data.shape[0]


class ShapeNet(Dataset):
    def __init__(self, num_points, split_num=3, partition='train'):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.num_points = num_points
        self.partition = partition        

        self.split_num = split_num
        self.max_distance = self.split_num + 1 
        
        load_partdata = True
        if load_partdata and os.path.exists('data/ShapeNetPart_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/ShapeNetPart_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance))
            self.part_distance = np.load(
                'data/ShapeNetPart_part_distance_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance))
        else:
            self.p2v_indices, self.part_distance, self.sigma_distance = split_part(self.data, self.split_num, self.max_distance)
            if not self.debug:
                np.save('data/ShapeNetPart_p2v_indices_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance),
                        self.p2v_indices)
                np.save('data/ShapeNetPart_part_distance_{}_splitnum_{}_md{}.npy'.format(partition, self.split_num, self.max_distance),
                        self.part_distance)
            print('Split part done!')
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        part_distance = self.part_distance[item]  # (1, 27, 27)
        p2v_indices = self.p2v_indices[item]  # (1, N)
        
        if self.partition == 'train':
            # shuffle pointcloud along N dim
            pc_rand_idx = np.arange(self.num_points)
            np.random.shuffle(pc_rand_idx)
            pointcloud = pointcloud[pc_rand_idx, :]
            p2v_indices = p2v_indices[pc_rand_idx]
            seg = seg[pc_rand_idx]

            # shuffle parts along N dim
            part_rand_idx = np.arange(self.split_num**3)                   
            np.random.shuffle(part_rand_idx)
            part_distance = part_distance[part_rand_idx][:, part_rand_idx]
        else:
            part_rand_idx = np.arange(self.split_num**3)

        return (pointcloud, label, p2v_indices, part_distance, part_rand_idx)

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN(Dataset):
    def __init__(self, num_points, root, partition='train', split_num=3, dataset='ScanObjectNN_hardest'):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        self.root = root
        self.split_num = split_num
        self.max_distance = self.split_num + 1
        self.dataset = dataset

        if self.dataset in ['ScanObjectNN_objectonly', 'ScanObjectNN_objectbg']:
            if self.partition == 'train':
                h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
                self.data = np.array(h5['data']).astype(np.float32)
                self.label = np.array(h5['label']).astype(int)
                h5.close()
            elif self.partition == 'test':
                h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
                self.data = np.array(h5['data']).astype(np.float32)
                self.label = np.array(h5['label']).astype(int)
                h5.close()
            else:
                raise NotImplementedError()
        elif self.dataset in ['ScanObjectNN_hardest']:
            if self.partition == 'train':
                h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
                self.data = np.array(h5['data']).astype(np.float32)
                self.label = np.array(h5['label']).astype(int)
                h5.close()
            elif self.partition == 'test':
                h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
                self.data = np.array(h5['data']).astype(np.float32)
                self.label = np.array(h5['label']).astype(int)
                h5.close()
            else:
                raise NotImplementedError()

        
        
        print(f'Successfully load ScanObjectNN shape of {self.data.shape}')
        
        self.data = np.array(self.data)
        self.label = self.label[:, np.newaxis]
        
        load_partdata = True
        if load_partdata and os.path.exists('data/ScanObjectNN_p2v_indices_{}_splitnum_{}_md{}_{}.npy'.format(partition, self.split_num, self.max_distance, self.dataset)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/ScanObjectNN_p2v_indices_{}_splitnum_{}_md{}_{}.npy'.format(partition, self.split_num, self.max_distance, self.dataset))
            self.part_distance = np.load(
                'data/ScanObjectNN_part_distance_{}_splitnum_{}_md{}_{}.npy'.format(partition, self.split_num, self.max_distance, self.dataset))
        else:
            self.p2v_indices, self.part_distance = split_part(self.data, self.split_num, self.max_distance)
            if not self.debug:
                np.save('data/ScanObjectNN_p2v_indices_{}_splitnum_{}_md{}_{}.npy'.format(partition, self.split_num, self.max_distance, self.dataset),
                        self.p2v_indices)
                np.save('data/ScanObjectNN_part_distance_{}_splitnum_{}_md{}_{}.npy'.format(partition, self.split_num, self.max_distance, self.dataset),
                        self.part_distance)
        
            print('Split part done!')
        
        


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        part_distance = self.part_distance[item]  # (1, 27, 27)
        p2v_indices = self.p2v_indices[item]  # (1, N)

            
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            
            # shuffle pointcloud along N dim
            pc_rand_idx = np.arange(self.num_points)
            np.random.shuffle(pc_rand_idx)
            pointcloud = pointcloud[pc_rand_idx, :]
            p2v_indices = p2v_indices[pc_rand_idx]

            # shuffle parts along N dim
            part_rand_idx = np.arange(self.split_num**3)          
            np.random.shuffle(part_rand_idx)
            part_distance = part_distance[part_rand_idx][:, part_rand_idx]
        else:
            part_rand_idx = np.arange(self.split_num**3)

        return (pointcloud, label, p2v_indices, part_distance, part_rand_idx)
    def __len__(self):
        return self.data.shape[0]