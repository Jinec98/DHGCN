from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ShapeNet
from model_unsup import DHGCN_DGCNN, DHGCN_AdaptConv
import numpy as np
from torch.utils.data import DataLoader
from util import  IOStream
import sklearn.metrics as metrics
from tqdm import tqdm

import time



def _init_():
    args.exp_name = args.exp_name + time.strftime("_%m_%d_%H_%M")

    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp train_pretrain.py checkpoints'+'/' +
              args.exp_name+'/'+'train_pretrain.py.backup')
    os.system('cp model_unsup.py checkpoints' + '/' +
              args.exp_name + '/' + 'model_unsup.py.backup')
    os.system('cp util.py checkpoints' + '/' +
              args.exp_name + '/' + 'util.py.backup')
    os.system('cp part_utils.py checkpoints' + '/' +
              args.exp_name + '/' + 'part_util.py.backup')
    os.system('cp data.py checkpoints' + '/' +
              args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    device = torch.device('cuda')
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, split_num=args.split_num),
                                num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, split_num=args.split_num),
                                num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'shapenet':
        train_loader = DataLoader(ShapeNet(partition='train', num_points=args.num_points, split_num=args.split_num),
                                num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ShapeNet(partition='test', num_points=args.num_points, split_num=args.split_num),
                                num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.backbone == 'AdaptConv':
        model = DHGCN_AdaptConv(args).to(device)
    elif args.backbone == 'DGCNN':
        model = DHGCN_DGCNN(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)


    # Train initalize
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100,
                        momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    best_test_acc = 0 
    scheduler = CosineAnnealingLR(opt, args.Tmax, eta_min=args.lr)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        if epoch < args.Tmax:
            scheduler.step()
        elif epoch == args.Tmax:
            for group in opt.param_groups:
                group['lr'] = 0.0001
        
        train_loss = 0.0
        train_loss_hop = 0.0
        count = 0.0
        model.train()
        train_pred_hop = []
        train_true_hop = []

        idx = 0
        total_time = 0.0
        for data, label, p2v_indices, part_distance, part_rand_idx in tqdm(train_loader):
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

            opt.zero_grad()
            start_time = time.time()
            _, hop_logits_list = model(data, p2v_indices, part_rand_idx)

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
            
            loss = hop_loss

            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            count += batch_size
            train_loss += loss.item() * batch_size
            train_loss_hop += hop_loss.item() * batch_size
            train_pred_hop.append(lasthop_logits.max(dim=1)[
                                  1].detach().cpu().numpy())
            train_true_hop.append(part_distance.cpu().numpy())

            idx += 1

        print('train total time is', total_time)
        train_true_hop = np.concatenate(train_true_hop).reshape(-1)
        train_pred_hop = np.concatenate(train_pred_hop).reshape(-1)  # (B,N)

        outstr = 'Train %d\nhop loss: %.6f, hop train acc: %.6f' % (
            epoch,
            train_loss*1.0 / count,
            metrics.accuracy_score(train_true_hop, train_pred_hop))  # Parthop train acc
        
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_loss_hop = 0.0
        count = 0.0
        model.eval()
        test_pred_hop = []
        test_true_hop = []

        total_time = 0.0
        for data, label, p2v_indices, part_distance, part_rand_idx in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            p2v_indices = p2v_indices.long().to(device)  # (B, N) #(B, 256)
            part_num = part_distance.shape[1]
            triu_idx = torch.triu_indices(part_num, part_num)

            # (B, 27, 27) -> (B, 27*26/2) 上三角
            part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
            part_distance = part_distance.long().to(device)
            part_rand_idx = part_rand_idx.to(device)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            opt.zero_grad()
            start_time = time.time()
            _, hop_logits_list = model(data, p2v_indices, part_rand_idx)

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

            loss = hop_loss

            end_time = time.time()
            total_time += (end_time - start_time)

            count += batch_size
            test_loss += loss.item() * batch_size
            test_loss_hop += hop_loss.item() * batch_size
            test_pred_hop.append(lasthop_logits.max(dim=1)[
                1].detach().cpu().numpy())
            test_true_hop.append(part_distance.cpu().numpy())

        print('test total time is', total_time)
        test_true_hop = np.concatenate(test_true_hop).reshape(-1)
        test_pred_hop = np.concatenate(test_pred_hop).reshape(-1)

        test_acc = metrics.accuracy_score(test_true_hop, test_pred_hop)

        outstr = 'Test %d\nhop loss: %.6f, hop test acc: %.6f' % (
            epoch,
            test_loss*1.0 / count,
            test_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(),
                       'checkpoints/%s/models/pretrain_model.t7' % args.exp_name)
        outstr = 'Best test hop acc: %.6f' % best_test_acc
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', default='DHGCN_pretrain', type=str, metavar='N', required=True,
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'shapenet'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--Tmax', type=int, default=250, metavar='N',
                        help='Max iteration number of scheduler. ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--gpu', type=str, default="0", metavar='N',
                        help='Cuda id')
    # ---------------------------------------------------------
    parser.add_argument('--backbone', type=str, choices=['AdaptConv', 'DGCNN'] , metavar='N', default='AdaptConv',   
                        help='choose backbone network')
    parser.add_argument('--split_num', type=int, default=3, metavar='N',
                        help='Voxel split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss function')    

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
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

