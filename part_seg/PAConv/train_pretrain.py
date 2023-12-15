from __future__ import print_function
import os
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from data import ShapeNetPart
import torch.nn.functional as F
from model_unsup import DHGCN_PAConv
import numpy as np
from torch.utils.data import DataLoader
from util import load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port
import logging
import random
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import sklearn.metrics as skmetrics
import time


classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    file_handler = logging.FileHandler(os.path.join('checkpoints', args.exp_name, 'main-' + str(int(time.time())) + '.log'))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger


def get_parser():
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--config', type=str, default='config/unsup_pretrain.yaml', help='config file')
    parser.add_argument('opts', help='see config/unsup_pretrain.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['sync_bn'] = cfg.get('sync_bn', True)
    cfg['dist_url'] = cfg.get('dist_url', 'tcp://127.0.0.1:6789')
    cfg['dist_backend'] = cfg.get('dist_backend', 'nccl')
    cfg['multiprocessing_distributed'] = cfg.get('multiprocessing_distributed', True)
    cfg['world_size'] = cfg.get('world_size', 1)
    cfg['rank'] = cfg.get('rank', 0)
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(gpu, ngpus_per_node):
    # ============= Model ===================
    num_part = 50
    model = DHGCN_PAConv(args, num_part)

    model.apply(weight_init)

    if main_process():
        logger.info(model)

    if args.sync_bn and args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    # =========== Dataloader =================
    train_data = ShapeNetPart(npoints=2048, split='trainval', normalize=False, split_num=args.split_num)
    if main_process():
        logger.info("The number of training data is:%d", len(train_data))

    test_data = ShapeNetPart(npoints=2048, split='test', normalize=False, split_num=args.split_num)
    if main_process():
        logger.info("The number of test data is:%d", len(test_data))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False, sampler=test_sampler)

    # ============= Optimizer ===================
    if args.use_sgd:
        if main_process():
            logger.info("Use SGD")
        opt = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        if main_process():
            logger.info("Use Adam")
        opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    if args.scheduler == 'cos':
        if main_process():
            logger.info("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    else:
        if main_process():
            logger.info("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0
    num_part = 50
    num_classes = 16

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes)

        test_metrics = test_epoch(test_loader, model, epoch, num_part, num_classes)

        # 1. when get the best hop accuracy, save the model:
        if test_metrics['accuracy'] > best_acc and main_process():
            best_acc = test_metrics['accuracy']
            logger.info('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/pretrain_model.pth' % args.exp_name)

    if main_process():
        logger.info('Final Max Hop Acc:%.5f' % best_acc)
        state = {
            'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_acc': best_acc}
        torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))



def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (cat_name, points, label, target, norm_plt, p2v_indices, part_distance, part_rand_idx) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        
        p2v_indices = p2v_indices.long().cuda(non_blocking=True)  # (B, N) #(B, 256)
        part_num = part_distance.shape[1]
        triu_idx = torch.triu_indices(part_num, part_num)
        # (B, 27, 27) -> (B, 27*26/2)
        part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
        part_distance = part_distance.long().cuda(non_blocking=True)
        part_rand_idx = part_rand_idx.cuda(non_blocking=True)
        
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        _, hop_logits_list = model(points, p2v_indices, part_rand_idx)  # seg_pred: b,n,50

        lasthop_logits = hop_logits_list[-1]  # (B,num_class, N,N)
        lasthop_logits = (
            lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
        # B,C,N,N -> (B,C,N*(N-1)/2)
        lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
        hop_loss = F.cross_entropy(
            lasthop_logits, part_distance)
        if True:
            for hop_logits in hop_logits_list[:-1]:
                hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                hop_loss += F.cross_entropy(hop_logits,
                                            part_distance)
            hop_loss /= len(hop_logits_list)

        loss = hop_loss
        
        # Loss backward
        if not args.multiprocessing_distributed:
            loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        train_true_hop = lasthop_logits.max(dim=1)[1].detach().cpu().numpy().reshape(-1)

        train_pred_hop = part_distance.cpu().numpy().reshape(-1)
        num_hop = len(train_true_hop)
        correct = torch.tensor(skmetrics.accuracy_score(train_true_hop, train_pred_hop, normalize=False)).to(loss.device)
        if args.multiprocessing_distributed:
            _count = lasthop_logits.new_tensor([batch_size], dtype=torch.long)   # same device with seg_pred!!!
            dist.all_reduce(loss)
            dist.all_reduce(_count)
            dist.all_reduce(correct)   # sum the correct across all processes
            # ! batch_size: the total number of samples in one iteration when with dist, equals to batch_size when without dist:
            batch_size = _count.item()
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
 
        accuracy.append(correct.item()/num_hop)   # append the accuracy of each iteration

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    if main_process():
        logger.info('Learning rate: %f', opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    outstr = 'Train %d, loss: %f, hop test acc: %f' % (epoch + 1, train_loss * 1.0 / count, metrics['accuracy'])

    if main_process():
        logger.info(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (cat_name, points, label, target, norm_plt, p2v_indices, part_distance, part_rand_idx) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        
        p2v_indices = p2v_indices.long().cuda(non_blocking=True)  # (B, N) #(B, 256)
        part_num = part_distance.shape[1]
        triu_idx = torch.triu_indices(part_num, part_num)
        # (B, 27, 27) -> (B, 27*26/2)
        part_distance = part_distance[:, triu_idx[0], triu_idx[1]]
        part_distance = part_distance.long().cuda(non_blocking=True)
        part_rand_idx = part_rand_idx.cuda(non_blocking=True)
        
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        
        _, hop_logits_list = model(points, p2v_indices, part_rand_idx)  # seg_pred: b,n,50

        lasthop_logits = hop_logits_list[-1]  # (B,num_class, N,N)
        lasthop_logits = (
            lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
        # B,C,N,N -> (B,C,N*(N-1)/2)
        lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
        hop_loss = F.cross_entropy(
            lasthop_logits, part_distance)
        if True:
            for hop_logits in hop_logits_list[:-1]:
                hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                hop_loss += F.cross_entropy(hop_logits,
                                            part_distance)
            hop_loss /= len(hop_logits_list)

        loss = hop_loss


        # accuracy:
        train_true_hop = lasthop_logits.max(dim=1)[1].detach().cpu().numpy().reshape(-1)
        train_pred_hop = part_distance.cpu().numpy().reshape(-1)
        correct = torch.tensor(skmetrics.accuracy_score(train_true_hop, train_pred_hop, normalize=False)).to(loss.device)
        num_hop = len(train_true_hop)
        if args.multiprocessing_distributed:
            _count = lasthop_logits.new_tensor([batch_size], dtype=torch.long)  # same device with seg_pred!!!
            dist.all_reduce(loss)
            dist.all_reduce(_count)
            dist.all_reduce(correct)  # sum the correct across all processes
            # ! batch_size: the total number of samples in one iteration when with dist, equals to batch_size when without dist:
            batch_size = _count.item()
        else:
            loss = torch.mean(loss)

        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item()/num_hop)  # append the accuracy of each iteration

    metrics['accuracy'] = np.mean(accuracy)

    outstr = 'Test %d, loss: %f, hop test acc: %f' % (epoch + 1, test_loss * 1.0 / count, metrics['accuracy'])

    if main_process():
        logger.info(outstr)

    return metrics


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    args.exp_name = time.strftime("%m_%d_%H_%M_") + args.exp_name 
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    if main_process():
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + args.exp_name):
            os.makedirs('checkpoints/' + args.exp_name)

        if not args.eval:  # backup the running files
            os.system('cp train_pretrain.py checkpoints' + '/' + args.exp_name + '/' + 'train_pretrain.py.backup')
            os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
            os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

        global logger
        logger = get_logger()
        logger.info(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert not args.eval, "The all_reduce function of PyTorch DDP will ignore/repeat inputs " \
                          "(leading to the wrong result), " \
                          "please use main.py to test (avoid DDP) for getting the right result."

    train(gpu, ngpus_per_node)


if __name__ == "__main__":
    args = get_parser()
    args.gpu = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.gpu)
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, args.ngpus_per_node, args)

