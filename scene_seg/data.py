import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from util import *
from part_utils import *
import json
import cv2  

def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1', split_num=3):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()
        
        self.split_num = split_num
        self.max_distance = self.split_num + 1  # 0,1,2,3
        
        load_partdata = True
        if load_partdata and os.path.exists('data/S3DIS_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num, self.max_distance, test_area)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/S3DIS_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num, self.max_distance, test_area))
            self.part_distance = np.load(
                'data/S3DIS_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num, self.max_distance, test_area))
        else:
            self.p2v_indices, self.part_distance = split_part(
                self.data, self.split_num, self.max_distance)  # p2v_indices: (B,N). part_distance: (B, 27, 27)
            if not self.debug:
                np.save('data/S3DIS_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num, self.max_distance, test_area),
                        self.p2v_indices)
                np.save('data/S3DIS_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num, self.max_distance, test_area),
                        self.part_distance)
            print('Split part done!')
        
    
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
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
            
        seg = torch.LongTensor(seg)

        return (pointcloud, seg, p2v_indices, part_distance, part_rand_idx)

    def __len__(self):
        return self.data.shape[0]


