import torch
import torch.nn as nn
import torch.nn.functional as F
from PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet
from cuda_lib.functional import assign_score_withk as assemble_dgcnn
from math import sqrt
import numpy as np


def gauss(x, sigma2=1, mu=0):
    return torch.exp(- (x-mu)**2 / (2*sigma2)) / (sqrt(2 * np.pi * sigma2))

def get_part_feature(x, p2v_indices, part_rand_idx, p2v_mask=None, part_num=27):
    """pool all point features in each voxel to generate part feature
    Args:
        x :(B, C, N)
        p2v_indices :(B, N)
        part_num (int, optional): _description_. Defaults to 27.
        part_rand_idx: (B, 27)
    Returns:
        _type_: _description_
    """
    B, C, N = x.size()

    if p2v_mask == None:
        # (B,N) -> (B,N,27)
        p2v_indices = p2v_indices.unsqueeze(-1).repeat(1, 1, part_num)
        # (B,N,27) == (B, 1, 27) -> bool: (B,N,27)
        p2v_mask = (p2v_indices == part_rand_idx.unsqueeze(1)).unsqueeze(1)
        # (B, 27)

    x = x.unsqueeze(-1).repeat(1, 1, 1, part_num)  # (B,C,N) -> (B,C,N, 27)

    # part_feature = (x * p2v_mask).max(2)[0]
    inpart_point_nums = p2v_mask.sum(2)  # (B,1,27)
    inpart_point_nums[inpart_point_nums == 0] = 1

    part_feature = (x * p2v_mask).sum(2) / inpart_point_nums
    # (B,C,N,27) * (B,N,27) -> (B,C,N,27) --sum-> (B,C,27)

    return part_feature, p2v_mask

def get_pointfeature_from_part(part_feature, p2v_mask, point_num=1024):
    """_summary_

    Args:
        part_feature (B,C,27):
        p2v_mask (B,N,27): 
    """
    part_feature = part_feature.unsqueeze(2).repeat(
        1, 1, point_num, 1)  # (B,C,27)->(B,C,N,27)
    part2point = (part_feature * p2v_mask).sum(-1)
    return part2point


def get_edge_feature(part_feature):
    """_summary_

    Args:
        part_feature (B, C, part_num)

    Returns:
        edge_feature (B, 2C, part_num, K i.e. part_num)
    """
    B, C, V = part_feature.shape
    part_feature = part_feature.transpose(
        2, 1).contiguous()  # (B, C, V) -> (B, N, C)

    edge_feature = part_feature.view(
        B, 1, V, C).repeat(1, V, 1, 1)  # (B,N,K=V,C)
    part_feature = part_feature.view(
        B, V, 1, C).repeat(1, 1, V, 1)  # (B,N,K,C)
    feature = torch.cat((edge_feature - part_feature, part_feature),
                        dim=3).permute(0, 3, 1, 2).contiguous()
    return feature



class HopGCN(nn.Module):
    def __init__(self, args, distance_numclass, in_channels, out_channels):
        super(HopGCN, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distance_numclass = distance_numclass

        self.edgeconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.hopconv = nn.Sequential(nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(out_channels/2)),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Conv2d(
                                         int(out_channels/2), distance_numclass, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(distance_numclass),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.hopmlp = nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.edgeconv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))

        # Graph attention
        self.num_head = 4
        assert out_channels % self.num_head == 0
        self.dim_per_head = out_channels // self.num_head
        self.atten1 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                        kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
        self.atten2 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                        kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))

    def forward(self, part_feature):
        B, C_in, N = part_feature.shape
        K = N

        edge_feature0 = get_edge_feature(part_feature)  # (B, 2C, N, K)

        edge_feature1 = self.edgeconv1(edge_feature0).view(B, self.dim_per_head,self.num_head, N, K)
        attention1 = self.atten1(edge_feature1)
        edge_feature1 = (attention1 * edge_feature1).view(B,-1,N,K)
        
        # Hop prediction
        hop_logits = self.hopconv(edge_feature1)
        hop = hop_logits.max(dim=1)[1]
        gauss_hop = gauss(hop, self.args.sigma2).view(B, 1,1, N, K)

        edge_feature2 = self.edgeconv2(edge_feature1).view(B, self.dim_per_head,self.num_head, N, K)
        attention2 = self.atten2(gauss_hop * edge_feature2)
        edge_feature2 = (attention2 * edge_feature2).view(B, -1, N, K)

        g = edge_feature2.mean(dim=-1, keepdim=False)  # (B, 256, N)
      
        return g, hop_logits


class DHGCN_PAConv(nn.Module):
    def __init__(self, args, num_part):
        super(DHGCN_PAConv, self).__init__()
        self.args = args
        self.num_part = num_part
        distance_numclass = self.args.split_num + 2

        self.k = args.get('k_neighbors', 30)
        self.calc_scores = args.get('calc_scores', 'softmax')
        self.hidden = args.get('hidden', [[16], [16], [16], [16]])  # the hidden layers of ScoreNet

        self.m2, self.m3, self.m4, self.m5 = args.get('num_matrices', [8, 8, 8, 8])
        self.scorenet2 = ScoreNet(10, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(10, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(10, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(10, self.m5, hidden_unit=self.hidden[3])

        i2 = 64  # channel dim of input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 64  # channel dim of output_3rd and input_4th
        o4 = i5 = 64  # channel dim of output_4th and input_5th
        o5 = 64  # channel dim of output_5th

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2 * 2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2 * 2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3 * 2, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3 * 2, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4 * 2, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m4 * o4)
        tensor5 = nn.init.kaiming_normal_(torch.empty(self.m5, i5 * 2, o5), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m5 * o5)

        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)
        self.matrice5 = nn.Parameter(tensor5, requires_grad=True)

        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(64, momentum=0.1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        # self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        # self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        # self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        # self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        # self.bn8 = nn.BatchNorm1d(128, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   nn.BatchNorm2d(64, momentum=0.1))

        # self.convt = nn.Sequential(nn.Conv1d(64*5, 1024, kernel_size=1, bias=False),
        #                            self.bnt,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.convc = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bnc,
        #                            nn.LeakyReLU(negative_slope=0.2))

        # self.conv6 = nn.Sequential(nn.Conv1d(1088+64*5, 256, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=args.get('dropout', 0.4))
        # self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=args.get('dropout', 0.4))
        # self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv9 = nn.Conv1d(128, num_part, kernel_size=1, bias=True)
        
 
        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                                nn.BatchNorm1d(64),
                                                nn.LeakyReLU(negative_slope=0.2))
        self.hopgcn1 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn2 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn3 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn4 = HopGCN(args, distance_numclass, 64*2, 64)

    def forward(self, x, p2v_indices, part_rand_idx):
        B, C, N = x.size()
        
        part_center, p2v_mask = get_part_feature(
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num**3)  # (B, 3, 27)
        part_center_embedding = self.conv_pcenteremb(part_center)
        
        idx, _ = knn(x, k=self.k)
        xyz = get_scorenet_input(x, k=self.k, idx=idx) 

        x = get_graph_feature(x, k=self.k, idx=idx)
        x = x.permute(0, 3, 1, 2)  # b,2cin,n,k
        x = self.relu1(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]

        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')
        x2 = self.relu2(self.bn2(x))
        part_feature1, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature1 = part_center_embedding + part_feature1  # ()
        g1, hop1_logits = self.hopgcn1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)
        x2 += part2point1
            
        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')
        x3 = self.relu3(self.bn3(x))
        part_feature2, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature2 = part_center_embedding + part_feature2  # ()
        g2, hop2_logits = self.hopgcn2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x3 += part2point2

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')
        x4 = self.relu4(self.bn4(x))
        part_feature3, _ = get_part_feature(
            x4, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3  # ()
        g3, hop3_logits = self.hopgcn3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x4 += part2point3     
        
        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')
        x5 = self.relu5(self.bn5(x))
        part_feature4, _ = get_part_feature(
            x5, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3  # ()
        g4, hop4_logits = self.hopgcn4(part_feature4)
        part2point4 = get_pointfeature_from_part(g4, p2v_mask, N)  # (B,C,N)
        x5 += part2point4
        
        xx = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # xc = self.convt(xx)
        # xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)

        # cls_label = cls_label.view(B, 16, 1)
        # cls_label = self.convc(cls_label)
        # cls = torch.cat((xc.view(B, 1024, 1), cls_label), dim=1)
        # cls = cls.repeat(1, 1, N)  # B,1088,N

        # x = torch.cat((xx, cls), dim=1)  # 1088+64*3
        # x = self.conv6(x)
        # x = self.dp1(x)
        # x = self.conv7(x)
        # x = self.dp2(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)  # b,n,50

        # if gt is not None:           
        #     return x, (hop1_logits, hop2_logits, hop3_logits, hop4_logits), F.nll_loss(x.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        # else:
        #     return x
        
        return xx, (hop1_logits, hop2_logits, hop3_logits, hop4_logits)


class ProjectionHead(nn.Module):
    def __init__(self, args, num_part):
        super(ProjectionHead, self).__init__()
        self.num_part = num_part
        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn8 = nn.BatchNorm1d(128, momentum=0.1)
        
        self.convt = nn.Sequential(nn.Conv1d(64*5, 1024, kernel_size=1, bias=False),
                                   self.bnt,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convc = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bnc,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(1088+64*5, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.get('dropout', 0.4))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.get('dropout', 0.4))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Conv1d(128, num_part, kernel_size=1, bias=True)

    def forward(self, x, cls_label, gt=None):
        B, C, N = x.size()
        xc = self.convt(x)
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)

        cls_label = cls_label.view(B, 16, 1)
        cls_label = self.convc(cls_label)
        cls = torch.cat((xc.view(B, 1024, 1), cls_label), dim=1)
        cls = cls.repeat(1, 1, N)  # B,1088,N

        x = torch.cat((x, cls), dim=1)  # 1088+64*3
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)
        x = self.dp2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # b,n,50

        if gt is not None:           
            return x, F.nll_loss(x.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return x
        
    
class LinearClassifier(nn.Module):
    def __init__(self, args, num_part):
        super(LinearClassifier, self).__init__()
        self.pretrain_model = DHGCN_PAConv(args, num_part)
        self.proj_head = ProjectionHead(args, num_part)
        

    def forward(self, x, norm_plt, cls_label, p2v_indices, part_rand_idx, gt=None):
        x, _ = self.pretrain_model(x, p2v_indices, part_rand_idx)
        
        x = self.proj_head(x, cls_label, gt)
        return x