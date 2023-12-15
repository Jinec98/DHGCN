from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS
from model import DHGCN_sceneseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

from tqdm import tqdm

import time

def _init_():
    args.exp_name = args.exp_name + time.strftime("_%m_%d_%H_%M")
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp train.py outputs'+'/' +
              args.exp_name+'/'+'train.py.backup')
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


def train(args, io):
    train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area, split_num=args.split_num), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area, split_num=args.split_num), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")


    model = DHGCN_sceneseg(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_loss_hop = 0.0
        train_loss_seg = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_pred_hop = []
        train_true_hop = []
        
        for data, seg, p2v_indices, part_distance, part_rand_idx in tqdm(train_loader):
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
            
            opt.zero_grad()
            seg_pred, hop_logits_list = model(data, p2v_indices, part_rand_idx)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            
            loss_seg = cal_loss(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            
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
                
                
            loss = loss_seg / loss_seg.detach() + hop_loss / hop_loss.detach()
            # print(loss_seg)
            
            loss.backward()
            opt.step()
            
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_loss_hop += hop_loss.item() * batch_size
            train_loss_seg += loss_seg.item() * batch_size
            
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_pred_hop.append(lasthop_logits.max(dim=1)[1].detach().cpu().numpy())
            train_true_hop.append(part_distance.cpu().numpy())
            
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
                    
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        train_true_hop = np.concatenate(train_true_hop).reshape(-1)
        train_pred_hop = np.concatenate(train_pred_hop).reshape(-1)  # (B,N)
        train_hop_acc = metrics.accuracy_score(train_true_hop, train_pred_hop)
        
        outstr = 'Train %d\nsegmentation loss:  %.6f, hop loss:  %.6f, multi-task loss: %.6f\ntrain acc: %.6f, train avg acc: %.6f, train iou: %.6f, hop train acc: %.6f' % (epoch, 
                                                                                                  train_loss_seg * 1.0 / count,
                                                                                                  train_loss_hop * 1.0 / count,
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious),
                                                                                                  train_hop_acc)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_loss_seg = 0.0
        test_loss_hop = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_pred_hop = []
        test_true_hop = []
        
        for data, seg, p2v_indices, part_distance, part_rand_idx in tqdm(test_loader):
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
            
            loss_seg = cal_loss(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            
            loss = loss_seg + hop_loss
            
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_loss_seg += loss_seg.item() * batch_size
            test_loss_hop += hop_loss.item() * batch_size
            
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_pred_hop.append(lasthop_logits.max(dim=1)[1].detach().cpu().numpy())
            test_true_hop.append(part_distance.cpu().numpy())
            
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        test_true_hop = np.concatenate(test_true_hop).reshape(-1)
        test_pred_hop = np.concatenate(test_pred_hop).reshape(-1)
        test_hop_acc = metrics.accuracy_score(test_true_hop, test_pred_hop)
        
        outstr = 'Test %d\nsegmentation loss: %.6f, hop loss: %.6f, multi-task loss: %.6f\ntest acc: %.6f, test avg acc: %.6f, test iou: %.6f, hop test acc: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_loss_seg * 1.0 / count,
                                                                                              test_loss_hop * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious),
                                                                                              test_hop_acc)
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)
        outstr = 'Best test iou: %.6f' % best_test_iou
        io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', default='DHGCN_train', type=str, metavar='N', required=True,
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--gpu', type=str, default="0", metavar='N',
                        help='Cuda id')
    # ---------------------------------------------------------
    parser.add_argument('--split_num', type=int, default=5, metavar='N',
                        help='Voxel split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel') 
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

    train(args, io)

