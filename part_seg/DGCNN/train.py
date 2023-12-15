import os
import sys
from util import *
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from manager import IouTable, get_miou
from data import ShapeNetPart, get_valid_labels
from importlib import import_module
from tqdm import tqdm
import time

TRAIN_NAME = __file__.split('.')[0]

class PartSegConfig():

    ####################
    # Dataset parameters
    ####################

    # Augmentations 
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    normal_scale = True
    augment_shift = None
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.25
    augment_noise = 0.002
    augment_noise_clip = 0.05
    augment_occlusion = 'none'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='DHGCN_train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='model', metavar='N',
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+',
                        help='set < 0 to use CPU')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--Tmax', type=int, default=100, metavar='N',
                        help='Max iteration number of scheduler. ')
    parser.add_argument('--mode', default= 'train', help= '[train/test]')
    parser.add_argument('--epoch', type= int, default= 200, help= 'Epoch number')
    parser.add_argument('--lr', type= float, default= 0.001, help= 'Learning rate')
    parser.add_argument('--bs', type= int, default= 32, help= 'Batch size')
    parser.add_argument('--dataset', type=str, default='data/shapenetcore_partanno_segmentation_benchmark_v0_normal', help= "Path to ShapeNetPart")
    parser.add_argument('--load', help= 'Path to load model')
    parser.add_argument('--record', type=str, default='record.log', help= 'Record file name (e.g. record.log)')
    parser.add_argument('--interval', type= int, default=100, help= 'Record interval within an epoch')
    parser.add_argument('--checkpoint_gap', type= int, default=10, help= 'Save checkpoints every n epochs')
    parser.add_argument('--point', type= int, default= 2048, help= 'Point number per object')
    # ---------------------------------------------------------
    parser.add_argument('--split_num', type=int, default=5, metavar='N',
                        help='Voxel split number')
    parser.add_argument('--single_hoploss', type=bool, default=False, metavar='N',
                        help='if only use the last hop loss')
    parser.add_argument('--sigma2', type=float, default=1.0, metavar='N',
                        help='sigma2 in gauss kernel') 
    
    args = parser.parse_args()

    if args.name == '':
        args.name = TRAIN_NAME
    args.name = args.name + time.strftime("_%m_%d_%H_%M")
    
    config = PartSegConfig()

    # Create Network
    MODEL = import_module(args.model)
    model = MODEL.DHGCN_DGCNN(args=args, class_num=50, cat_num=16)
    manager = Manager(model, args)

    ################
    # Start Training
    ################

    print("Training ...")
    train_data = ShapeNetPart(root=args.dataset, config=config, num_points=args.point, split='trainval', split_num=args.split_num)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.bs, drop_last=True)
    test_data = ShapeNetPart(root=args.dataset, config=config, num_points=args.point, split='test', split_num=args.split_num)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.bs, drop_last=False)
    manager.train(train_loader, test_loader)
        

class Manager():
    def __init__(self, model, args):

        ############
        # Parameters
        ############
        self.args = args
        self.args_info = args.__str__()
        self.device = torch.device('cpu' if len(args.gpu) == 0 else 'cuda:{}'.format(args.gpu[0]))
        self.model = model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=args.gpu)
        print('Now use {} GPUs: {}'.format(len(args.gpu), args.gpu))
        if args.load:
            self.model.load_state_dict(torch.load(args.load))
        
        self.epoch = args.epoch
        self.Tmax = args.Tmax
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.Tmax, eta_min=args.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.save = os.path.join('models', args.name, 'checkpoints')
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        self.record_interval = args.interval
        self.record_file = None
        if args.record:
            self.record_file = open(os.path.join('models', args.name, args.record), 'w')
        self.checkpoint_gap = args.checkpoint_gap

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

    def train(self, train_data, test_data):
        self.record("*****************************************")
        self.record("Hyper-parameters: {}".format(self.args_info))
        self.record("Model parameter number: {}".format(parameter_number(self.model)))
        self.record("Model structure: \n{}".format(self.model.__str__()))
        self.record("*****************************************")
        
        best_c_miou = 0.0
        best_i_miou = 0.0

        
        for epoch in range(self.epoch):
            self.model.train()
            train_loss = 0.0
            train_loss_hop = 0.0
            train_loss_seg = 0.0
            train_iou_table = IouTable()
            learning_rate = self.optimizer.param_groups[0]['lr']
            for i, (cat_name, obj_ids, points, labels, mask, onehot, p2v_indices, part_distance, part_rand_idx) in enumerate(tqdm(train_data)):
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
                
                out, hop_logits_list = self.model(points, onehot, p2v_indices, part_rand_idx)
                
                self.optimizer.zero_grad()
                loss_seg = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1,))  
                
                lasthop_logits = hop_logits_list[-1]  # (B,num_class, N,N)
                lasthop_logits = (
                    lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
                # B,C,N,N -> (B,C,N*(N-1)/2)
                lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
                hop_loss = F.cross_entropy(
                    lasthop_logits, part_distance)
                if not self.args.single_hoploss:
                    for hop_logits in hop_logits_list[:-1]:
                        hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                        hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                        hop_loss += F.cross_entropy(hop_logits,
                                                    part_distance)
                    hop_loss /= len(hop_logits_list)
                    
                    
                loss = loss_seg / loss_seg.detach() + hop_loss / hop_loss.detach()
                
                   
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_loss_hop += hop_loss.item()
                train_loss_seg += loss_seg.item()
                
                out[mask == 0] = out.min()
                pred = torch.max(out, 2)[1]
                self.calculate_save_mious(train_iou_table, cat_name, labels, pred)

                # record within epoch
                if self.record_interval and ((i + 1) % self.record_interval == 0):
                    c_miou = train_iou_table.get_mean_category_miou()
                    i_miou = train_iou_table.get_mean_instance_miou()
                    self.record(' epoch {:3} step {:5} | avg loss: {:.3f} seg loss: {:.3f} hop loss: {:.3f} | miou(c): {:.3f} | miou(i): {:.3f}'.format(epoch+1, i+1, train_loss/(i + 1), train_loss_seg/(i + 1), train_loss_hop/(i + 1), c_miou, i_miou))

            train_loss /= (i+1) 
            train_loss_hop /= (i+1)
            train_loss_seg /= (i+1)
            train_table_str = train_iou_table.get_string()
            
            # test
            test_loss, test_table_str, best_c_miou, best_i_miou = self.test(test_data, best_c_miou, best_i_miou)
            
            if epoch < self.Tmax:
                self.lr_scheduler.step()
            elif epoch == self.Tmax:
                for group in self.optimizer.param_groups:
                    group['lr'] = 0.0001

            # save checkpoints
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save, 'model.pkl'))
                # Save checkpoints occasionally
                if (epoch+1) % self.checkpoint_gap == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save, 'epoch_{:03d}.pkl'.format(epoch)))

            # Record IoU
            self.record("==== Epoch {:3} ====".format(epoch + 1))
            self.record("Training mIoU:")
            self.record(train_table_str)
            self.record("Testing mIoU:")
            self.record(test_table_str)
            
            self.record("Best class mIoU:")
            self.record(str(best_c_miou))
            self.record("Best instance mIoU:")
            self.record(str(best_i_miou))


    def test(self, test_data, best_c_miou, best_i_miou):
        self.model.eval()
        test_loss = 0
        test_loss_hop = 0.0
        test_loss_seg = 0.0
        test_iou_table = IouTable()
    
        for i, (cat_name, obj_ids, points, labels, mask, onehot, p2v_indices, part_distance, part_rand_idx) in enumerate(tqdm(test_data)):
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
                
            loss_seg = self.loss_function(out.reshape(-1, out.size(-1)), labels.view(-1,))  
            lasthop_logits = hop_logits_list[-1]  # (B,num_class, N,N)
            lasthop_logits = (
                lasthop_logits + lasthop_logits.permute(0, 1, 3, 2)) / 2
            # B,C,N,N -> (B,C,N*(N-1)/2)
            lasthop_logits = lasthop_logits[:, :, triu_idx[0], triu_idx[1]]
            hop_loss = F.cross_entropy(
                lasthop_logits, part_distance)
            if not self.args.single_hoploss:
                for hop_logits in hop_logits_list[:-1]:
                    hop_logits = (hop_logits + hop_logits.permute(0, 1, 3, 2)) / 2
                    hop_logits = hop_logits[:, :, triu_idx[0], triu_idx[1]]
                    hop_loss += F.cross_entropy(hop_logits,
                                                part_distance)
                hop_loss /= len(hop_logits_list)
                    
                    
            loss = loss_seg / loss_seg.detach() + hop_loss / hop_loss.detach()
            
            test_loss += loss.item()
            test_loss_hop += hop_loss.item()
            test_loss_seg += loss_seg.item()
            
            out[mask == 0] = out.min()
            pred = torch.max(out, 2)[1]
            self.calculate_save_mious(test_iou_table, cat_name, labels, pred)

        test_loss /= (i+1) 
        test_loss_hop /= (i+1) 
        test_loss_seg /= (i+1) 
        
        c_miou = test_iou_table.get_mean_category_miou()
        i_miou = test_iou_table.get_mean_instance_miou()
        
        if best_c_miou < c_miou:
            best_c_miou = c_miou
            torch.save(self.model.state_dict(), os.path.join(self.save, 'model_best_c_miou.pkl'))
        if best_i_miou < i_miou:
            best_i_miou = i_miou
            torch.save(self.model.state_dict(), os.path.join(self.save, 'model_best_i_miou.pkl'))
        
        test_table_str = test_iou_table.get_string()

        return test_loss, test_table_str, best_c_miou, best_i_miou


if __name__ == '__main__':
    main()
