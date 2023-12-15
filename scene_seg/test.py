from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from data import S3DIS
from model import DHGCN_sceneseg
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement

import time

global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True


def _init_():
    args.exp_name = args.exp_name + time.strftime("_%m_%d_%H_%M")
  
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp test.py outputs'+'/' +
              args.exp_name+'/'+'test.py.backup')
    os.system('cp model.py outputs' + '/' +
              args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' +
              args.exp_name + '/' + 'util.py.backup')
    os.system('cp part_utils.py outputs' + '/' +
              args.exp_name + '/' + 'part_util.py.backup')
    os.system('cp data.py outputs' + '/' +
              args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(13):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all 


def visualization(visu, visu_format, test_choice, data, seg, pred, visual_file_index, semseg_colors):
    global room_seg, room_pred
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = [] 
        skip = False
        with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index+1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True 
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False  
            else:
                skip = True
        elif test_choice !='all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname)
            
            data = np.loadtxt('data/indoor3d_sem_seg_hdf5_data_test/raw_data3d/Area_'+test_area+'/'+roomname+'('+str(visual_file_index)+').txt')
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB.append(semseg_colors[int(pred[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:,[1,2,0]]
            xyzRGB = np.concatenate((data, np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_seg.append(seg[i].cpu().numpy())
            room_pred.append(pred[i].cpu().numpy()) 
            f = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt', "a")
            f_gt = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt', "a")
            np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ') 
            np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
            
            if roomname != roomname_next:
                mIoU = np.mean(calculate_sem_IoU(np.array(room_pred), np.array(room_seg), visual=True))
                mIoU = str(round(mIoU, 4))
                room_pred = []
                room_seg = []
                if visu_format == 'ply':
                    filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.ply'
                    filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.ply'
                    xyzRGB = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt')
                    xyzRGB_gt = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt')
                    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath)
                    print('PLY visualization file saved in', filepath)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system('rm -rf '+'outputs/'+args.exp_name+'/visualization/area_'+test_area+'/'+roomname+'/*.txt')
                else:
                    filename = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt'
                    filename_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt'
                    filename_mIoU = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.txt'
                    os.rename(filename, filename_mIoU)
                    print('TXT visualization file saved in', filename_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            

def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        visual_file_index = 0
        test_area = str(test_area)
        if os.path.exists("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
                    visual_file_index = visual_file_index + 1
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area, split_num=args.split_num),
                                     batch_size=args.batch_size, shuffle=False, drop_last=False)

            device = torch.device("cuda" if args.cuda else "cpu")
                        
            #Try to load models
            semseg_colors = test_loader.dataset.semseg_colors            
            model = DHGCN_sceneseg(args).to(device)
            model = nn.DataParallel(model)
                        
            model.load_state_dict(torch.load(os.path.join(args.model_path, 'model_%s.t7' % test_area)))
            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg, p2v_indices, part_distance, part_rand_idx in test_loader:
                data, seg = data.to(device), seg.to(device)
                p2v_indices = p2v_indices.long().to(device)  # (B, N) #(B, 256)
                part_num = part_distance.shape[1]
                triu_idx = torch.triu_indices(part_num, part_num)

                # (B, 27, 27) -> (B, 27*26/2)
                part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
                part_distance = part_distance.long().to(device)
                part_rand_idx = part_rand_idx.to(device)
                
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                
                seg_pred, hop_logits_list = model(data, p2v_indices, part_rand_idx)               
                
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1] 
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # visiualization
                visualization(args.visu, args.visu_format, args.test_area, data, seg, pred, visual_file_index, semseg_colors) 
                visual_file_index = visual_file_index + data.shape[0]
            if visual_warning and args.visu != '':
                print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', default='DHGCN_test', type=str, metavar='N', required=True,
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    parser.add_argument('--gpu', type=str, default="0", metavar='N',
                        help='Cuda id')
    # ---------------------------------------------------------
    parser.add_argument('--split_num', type=int, default=5, metavar='N',
                        help='Voxel split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel') 
    parser.add_argument('--model_path', type=str, default='models/', metavar='N',
                    help='Pretrained model path')
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    test(args, io)
