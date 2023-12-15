from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import ModelNet40
from model import DHGCN_DGCCN, DHGCN_AdaptConv, DHGCN_PRANet
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss
import sklearn.metrics as metrics
from tqdm import tqdm
import time


def test(args):
    device = torch.device('cuda')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, split_num=args.split_num),
                            num_workers=8,
                            batch_size=args.batch_size, shuffle=False, drop_last=False)
  
    if args.backbone == 'DGCNN':
        model = DHGCN_DGCCN(args).to(device)
    elif args.backbone == 'AdaptConv':
        model = DHGCN_AdaptConv(args).to(device)
    elif args.backbone == 'PRANet':
        model = DHGCN_PRANet(args).to(device)
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
    test_loss_hop = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    test_pred_hop = []
    test_true_hop = []

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
        logits, hop_logits_list = model(data, p2v_indices, part_rand_idx)

        cls_loss = cal_loss(logits, label)

        lasthop_logits = hop_logits_list[-1]  # (B,num_class, N,N)
        lasthop_logits = (
            lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
        # B,C,N,N -> (B,C,N*(N-1)/2)
        lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
        hop_loss = F.cross_entropy(
            lasthop_logits, part_distance, label_smoothing=0.2)
        if not args.single_hoploss:
            for hop_logits in hop_logits_list[:-1]:
                hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                hop_loss += F.cross_entropy(hop_logits,
                                            part_distance, label_smoothing=0.2)
            hop_loss /= len(hop_logits_list)

        loss = cls_loss + hop_loss

        end_time = time.time()
        total_time += (end_time - start_time)

        preds = logits.max(dim=1)[1]

        count += batch_size
        test_loss += loss.item() * batch_size
        test_loss_cls += cls_loss.item() * batch_size
        test_loss_hop += hop_loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        test_pred_hop.append(lasthop_logits.max(dim=1)[1].detach().cpu().numpy())
        test_true_hop.append(part_distance.cpu().numpy())

    print('test total time is', total_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_true_hop = np.concatenate(test_true_hop).reshape(-1)
    test_pred_hop = np.concatenate(test_pred_hop).reshape(-1)

    test_acc = metrics.accuracy_score(test_true, test_pred)

    outstr = 'Test:: \nclassification loss: %.6f, hop loss: %.6f, multi-task loss: %.6f\nClassification test acc: %.6f, test avg acc: %.6f, Hop test acc: %.6f' % (
        test_loss_cls * 1.0 / count,
        test_loss_hop * 1.0 / count,
        test_loss*1.0 / count,
        test_acc,
        metrics.balanced_accuracy_score(
            test_true, test_pred), 
        metrics.accuracy_score(test_true_hop, test_pred_hop)) 
    print(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', default='DHGCN_test', type=str, metavar='N', required=False,
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--gpu', type=str, default="0", metavar='N',
                        help='Cuda id')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in backbone')
    parser.add_argument('--model_path', type=str, default='./models/model_adapt.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--resume', type=bool, default=True, metavar='N',
                        help='Restore model from path')
    # ---------------------------------------------------------
    parser.add_argument('--backbone', type=str, default='AdaptConv', metavar='N',
                        choices=['DGCNN', 'AdaptConv', 'PRANet']) 
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
