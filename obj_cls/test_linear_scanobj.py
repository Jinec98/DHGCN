from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import ScanObjectNN
from model_unsup import LinearClassifier
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss
import sklearn.metrics as metrics
from tqdm import tqdm
import time


def test(args):
    device = torch.device('cuda')
    if args.dataset == 'ScanObjectNN_objectonly':
        test_loader = DataLoader(ScanObjectNN(root='data/ScanObjectNN/main_split_nobg', partition='test', num_points=args.num_points, split_num=args.split_num, dataset=args.dataset), num_workers=8,
                                 batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN_objectbg':
        test_loader = DataLoader(ScanObjectNN(root='data/ScanObjectNN/main_split',partition='test', num_points=args.num_points, split_num=args.split_num, dataset=args.dataset), num_workers=8,
                                 batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN_hardest':
        test_loader = DataLoader(ScanObjectNN(root='data/ScanObjectNN/main_split', partition='test', num_points=args.num_points, split_num=args.split_num, dataset=args.dataset), num_workers=8,
                                 batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LinearClassifier(args, output_channels=15).to(device)
    print(str(model))
    model = nn.DataParallel(model)
    
    if args.resume:
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Loaded model: %s" % (args.model_path))
    
    ####################
    # Test
    ####################
    test_loss = 0.0
    test_loss_cls = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []

    total_time = 0.0
    
    for data, label, p2v_indices, part_distance, part_rand_idx in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        p2v_indices = p2v_indices.long().to(device)  # (B, N) #(B, 256)
        part_num = part_distance.shape[1]
        triu_idx = torch.triu_indices(part_num, part_num)

        # (B, 27, 27) -> (B, 27*26/2) 
        part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
        part_distance = part_distance.long().to(device)
        part_rand_idx = part_rand_idx.to(device)

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        start_time = time.time()
        logits = model(data, p2v_indices, part_rand_idx)

        cls_loss = cal_loss(logits, label)     
        loss = cls_loss

        end_time = time.time()
        total_time += (end_time - start_time)

        preds = logits.max(dim=1)[1]

        count += batch_size
        test_loss += loss.item() * batch_size
        test_loss_cls += cls_loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    print('test total time is', total_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_acc = metrics.accuracy_score(test_true, test_pred)

    outstr = 'Test::\nloss: %.6f, acc: %.6f' % (
        test_loss*1.0 / count,
        test_acc)
    print(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', default='DHGCN_unsupervised_test_scanobj', type=str, metavar='N', required=False,
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='ScanObjectNN_objectbg', metavar='N',
                        choices=['ScanObjectNN_objectonly', 'ScanObjectNN_objectbg', 'ScanObjectNN_hardest'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--gpu', type=str, default="0", metavar='N',
                        help='Cuda id')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in backbone')
    parser.add_argument('--model_path', type=str, default='./models/model_linear_scanobj_bg.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--resume', type=bool, default=True, metavar='N',
                        help='Restore model from path')
    # ---------------------------------------------------------
    parser.add_argument('--backbone', type=str, default='DGCNN', metavar='N',
                        choices=['DGCNN']) 
    parser.add_argument('--split_num', type=int, default=3, metavar='N',
                        help='Voxelization split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel') 

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    test(args)
