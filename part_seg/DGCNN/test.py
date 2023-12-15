import os
import sys
from util import *
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from manager import IouTable, get_miou
from data import ShapeNetPart, get_valid_labels
from importlib import import_module
import json
from tqdm import tqdm

TRAIN_NAME = __file__.split('.')[0]

color_map = json.load(open('./meta/part_color_mapping.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='DHGCN_test', help='Name of the experiment')
parser.add_argument('--model', type=str, default='model', help='Model to use')
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='set < 0 to use CPU')
parser.add_argument('--bs', type= int, default=16, help= 'Batch size')
parser.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--dataset', type=str, default='data/shapenetcore_partanno_segmentation_benchmark_v0_normal', help= "Path to ShapeNetPart")
parser.add_argument('--model_path', default='DHGCN_train', help= 'Path to pretrained model')
parser.add_argument('--checkpoint', default='model_best_c_miou.pkl', help= 'Model to test')
parser.add_argument('--record', type=str, default='record.log', help= 'Record file name (e.g. record.log)')
parser.add_argument('--point', type= int, default= 2048, help= '')
parser.add_argument('--vis_dir', help= 'Folder for output visualization')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
# ---------------------------------------------------------
parser.add_argument('--split_num', type=int, default=5, metavar='N',
                    help='Voxel split number')
parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                    help='if only use the last hop loss')
parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel') 
args = parser.parse_args()


def main():

    # Create Network
    MODEL = import_module(args.model)
    model = MODEL.DHGCN_DGCNN(args=args, class_num=50, cat_num=16)

    # Dataset
    test_data = ShapeNetPart(root=args.dataset, config=None, num_points=args.point, split='test',
                                split_num=args.split_num)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.bs, drop_last=False)
    if args.vis_dir:
        for cat in sorted(test_data.seg_classes.keys()):
            if not os.path.exists(os.path.join(args.vis_dir, cat)):
                os.makedirs(os.path.join(args.vis_dir, cat))

    # Testing
    manager = Manager(model, args)
    manager.test(test_loader)


class Manager():
    def __init__(self, model, args):

        ############
        # Parameters
        ############

        self.bs = args.bs
        self.args_info = args.__str__()
        self.device = torch.device('cpu' if len(args.gpu) == 0 else 'cuda:{}'.format(args.gpu[0]))
        self.model = model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=args.gpu)
        print('Now use {} GPUs: {}'.format(len(args.gpu), args.gpu))

        # load pretrained model
        self.load = os.path.join('models', args.model_path, 'checkpoints', args.checkpoint)
        self.model.load_state_dict(torch.load(self.load, map_location='cpu'))        
        
        print('Load pretrained model from: {}'.format(self.load))

        # path to save
        if args.name == '':
            self.save = os.path.join('test', args.model_path)
        else:
            self.save = os.path.join('test', args.name)
        self.save = os.path.join(self.save, os.path.splitext(args.checkpoint)[0])

        if not os.path.exists(self.save):
            os.makedirs(self.save)
        self.record_file = None
        if args.record:
            self.record_file = open(os.path.join(self.save, args.record), 'w')

        # path to vis_dir
        self.outdir = None
        if args.vis_dir:
            self.outdir = args.vis_dir


    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
            self.record_file.flush()

    def calculate_save_mious(self, iou_table, category_names, labels, predictions):
        for i in range(len(category_names)):
            category = category_names[i]
            pred = predictions[i]
            label =  labels[i]
            valid_labels = get_valid_labels(category)
            miou = get_miou(pred, label, valid_labels)
            iou_table.add_obj_miou(category, miou)

    def test(self, test_data):
        ############
        # Test Model
        ############

        # record arguments
        self.record(self.args_info)
        
        self.model.eval()
        test_iou_table = IouTable()

        for i, (
                cat_name, obj_ids, points, labels, mask, onehot, p2v_indices, part_distance,
                part_rand_idx) in enumerate(
            tqdm(test_data)):
            points = points.to(self.device)
            labels = labels.to(self.device)
            onehot = onehot.to(self.device)
            p2v_indices = p2v_indices.long().to(self.device)  # (B, N) #(B, 256)
            part_num = part_distance.shape[1]
            triu_idx = torch.triu_indices(part_num, part_num)
            # (B, 27, 27) -> (B, 27*26/2)
            part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
            part_distance = part_distance.long().to(self.device)
            part_rand_idx = part_rand_idx.to(self.device)
            with torch.no_grad():
                out, hop_logits_list = self.model(points, onehot, p2v_indices, part_rand_idx)

            out[mask == 0] = out.min()
            pred = torch.max(out, 2)[1]

            # compute iou
            self.calculate_save_mious(test_iou_table, cat_name, labels, pred)

            # output obj files
            if self.outdir:
                pred = pred.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                pts = points.detach().cpu().numpy()[:,:,0:3]
                for shape_idx in range(pred.shape[0]):
                    obj_idx = shape_idx+i*self.bs
                    output_color_point_cloud(pts[shape_idx], labels[shape_idx], 
                        os.path.join(self.outdir, cat_name[shape_idx], str(obj_idx)+'_gt.obj'))
                    output_color_point_cloud(pts[shape_idx], pred[shape_idx], 
                        os.path.join(self.outdir, cat_name[shape_idx], str(obj_idx)+'_pred.obj'))
                    output_color_point_cloud_red_blue(pts[shape_idx], np.int32(labels[shape_idx] == pred[shape_idx]), 
                        os.path.join(self.outdir, cat_name[shape_idx], str(obj_idx)+'_diff.obj'))

        self.record(test_iou_table.get_string())


def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            
if __name__ == '__main__':
    main()
