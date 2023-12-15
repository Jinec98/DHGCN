from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
import torch.nn.functional as F
import torch.nn as nn
from model_unsup import LinearClassifier
import numpy as np
from torch.utils.data import DataLoader
from util import to_categorical, compute_overall_iou, load_cfg_from_cfg_file, merge_cfg_from_list, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import json

classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']
color_map = json.load(open('meta/part_color_mapping.json', 'r'))
color_map = np.array(color_map)

def get_parser():
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--config', type=str, default='config/test.yaml', help='config file')
    parser.add_argument('opts', help='see config/test.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        data = pc_normalize(data)
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        data = pc_normalize(data)
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            

def test(args, io):
    # Dataloader
    test_data = ShapeNetPart(npoints=2048, split='test', normalize=False, split_num=args.split_num)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)
    
    if args.vis_dir:
        for cat in sorted(test_data.seg_classes.keys()):
            if not os.path.exists(os.path.join(args.vis_dir, cat)):
                os.makedirs(os.path.join(args.vis_dir, cat))

    # Try to load models
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    model = LinearClassifier(args, num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("models/%s/best_%s_model.pth" % (args.model_name, args.model_type),
                            map_location=torch.device('cpu'))
    
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 50
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

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
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes), p2v_indices, part_rand_idx)  # b,n,50

        # output obj files
        if args.vis_dir:
            pred = torch.max(seg_pred, 2)[1]
            pred = pred.detach().cpu().numpy()
            labels = target.detach().cpu().numpy()
            pts = points.detach().transpose(2, 1).cpu().numpy()[:,:,0:3]
            for shape_idx in range(pred.shape[0]):
                obj_idx = shape_idx+batch_id*batch_size
                output_color_point_cloud(pts[shape_idx], labels[shape_idx], 
                    os.path.join(args.vis_dir, cat_name[shape_idx], str(obj_idx)+'_gt.obj'))
                output_color_point_cloud(pts[shape_idx], pred[shape_idx], 
                    os.path.join(args.vis_dir, cat_name[shape_idx], str(obj_idx)+'_pred.obj'))
                output_color_point_cloud_red_blue(pts[shape_idx], np.int32(labels[shape_idx] == pred[shape_idx]), 
                    os.path.join(args.vis_dir, cat_name[shape_idx], str(obj_idx)+'_diff.obj'))


        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(16):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


if __name__ == "__main__":
    args = get_parser()
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    test(args, io)

