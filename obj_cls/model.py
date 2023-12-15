import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# for PRA-Net
from pranet_lib.intra_structure import ISL
from pranet_lib.inter_region import IRL


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def gauss(x, sigma2=1, mu=0):
    return torch.exp(- (x-mu)**2 / (2*sigma2)) / (sqrt(2 * torch.pi * sigma2))


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


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


class AdaptiveConv(nn.Module):
    def __init__(self, x_channels, out_channels, in_channels):
        super(AdaptiveConv, self).__init__()
        self.x_channels = x_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv0 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(
            out_channels, out_channels*x_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y)  # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y)  # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k,
                                       self.out_channels, self.x_channels)  # (bs, num_points, k, out, in)

        # (bs, num_points, k, in_channels, 1)
        x = x.permute(0, 2, 3, 1).unsqueeze(4)
        x = torch.matmul(y, x).squeeze(4)  # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)  # (32, 64, 1024, 20) # (B, C, N, K)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B,C,N)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (B, 256, N, K)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x = self.conv5(x)  # fc 512->1024
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


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


class DHGCN_DGCCN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DHGCN_DGCCN, self).__init__()
        self.args = args
        self.k = args.k
        distance_numclass = self.args.split_num + 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        # DGCNN backbone
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # HopGCN
        self.hopgcn1 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn2 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn3 = HopGCN(args, distance_numclass, 128*2, 128)
        self.hopgcn4 = HopGCN(args, distance_numclass, 256*2, 256)
        
        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                                 nn.BatchNorm1d(64),
                                                 nn.LeakyReLU(negative_slope=0.2))

        # Projection head
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, p2v_indices, part_rand_idx):
        batch_size, C, N = x.shape

        part_center, p2v_mask = get_part_feature(
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num**3)  # (B, 3, 27)

        # Layer 1
        x, _ = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_center_embedding = self.conv_pcenteremb(part_center)
        part_feature1 = part_center_embedding + part_feature1  # ()
        g1, hop1_logits = self.hopgcn1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)
        x1 += part2point1

        # Layer 2
        x, _ = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g2, hop2_logits = self.hopgcn2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2

        # Layer 3
        x, _ = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g3, hop3_logits = self.hopgcn3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3

        # Layer 4
        x, _ = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)  # ()
        x4 = x.max(dim=-1, keepdim=False)[0]
        part_feature4, _ = get_part_feature(
            x4, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g4, hop4_logits = self.hopgcn4(part_feature4)
        part2point4 = get_pointfeature_from_part(g4, p2v_mask, N)  # (B,C,N)
        x4 += part2point4

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)  # point-wise feature (B,C, N)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, (hop1_logits, hop2_logits, hop3_logits, hop4_logits)
    

class DHGCN_AdaptConv(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DHGCN_AdaptConv, self).__init__()
        self.args = args
        self.k = 10
        distance_numclass = self.args.split_num + 2
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # AdaptConv backbone
        self.adapt_conv1 = AdaptiveConv(6, 64, 6)
        self.adapt_conv2 = AdaptiveConv(6, 64, 64*2)
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # HopGCN
        self.hopgcn1 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn2 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn3 = HopGCN(args, distance_numclass, 128*2, 128)
        self.hopgcn4 = HopGCN(args, distance_numclass, 256*2, 256)
        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                                 nn.BatchNorm1d(64),
                                                 nn.LeakyReLU(negative_slope=0.2))
            
        # Projection head
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, p2v_indices, part_rand_idx):
        batch_size, C, N = x.shape
        points = x

        part_center, p2v_mask = get_part_feature(
            points, p2v_indices, part_rand_idx, part_num=self.args.split_num**3)  # (B, 3, 27)

        # Layer 1
        x, idx = get_graph_feature(x, k=self.k)
        p, _ = get_graph_feature(points, k=self.k, idx=idx)
        x = self.adapt_conv1(p, x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        part_center_embedding = self.conv_pcenteremb(part_center)
        part_feature1 = part_center_embedding + part_feature1
        g1, hop1_logits = self.hopgcn1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)
        x1 += part2point1

        # Layer 2
        x, idx = get_graph_feature(x1, k=self.k)
        p, _ = get_graph_feature(points, k=self.k, idx=idx)
        x = self.adapt_conv2(p, x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g2, hop2_logits = self.hopgcn2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2

        # Layer 3
        x, _ = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g3, hop3_logits = self.hopgcn3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3

        # Layer 4
        x, _ = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        part_feature4, _ = get_part_feature(
            x4, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g4, hop4_logits = self.hopgcn4(part_feature4)
        part2point4 = get_pointfeature_from_part(g4, p2v_mask, N)  # (B,C,N)
        x4 += part2point4

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (hop1_logits, hop2_logits, hop3_logits, hop4_logits)
    
    
class DHGCN_PRANet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DHGCN_PRANet, self).__init__()
        self.args = args
        self.k_hat = 20
        distance_numclass = self.args.split_num + 2
        
        self.sample_ratio = [0, 4, 8, 16]
        self.m_list = [0,4,4,4]
        self.start_layer = 1
        self.init = True
        
        # ISL layers
        self.ISL1 = ISL(in_channel=3 * 2, out_channel_list=[64], k_hat=self.k_hat, bias=False)
        self.ISL2 = ISL(in_channel=64 * 2, out_channel_list=[64], k_hat=self.k_hat, bias=False)
        self.ISL3 = ISL(in_channel=64 * 2, out_channel_list=[128], k_hat=self.k_hat, bias=False)
        self.ISL4 = ISL(in_channel=128 * 2, out_channel_list=[256], k_hat=self.k_hat, bias=False)

        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        # FC layers
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)

        self.bn18 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn19 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        # IRL layers
        channels = [64, 64, 128, 256]
        
        self.irl_list = nn.ModuleList()
        for i in range(self.start_layer, 4):
            irl = IRL(channel=channels[i], sample_ratio=self.sample_ratio[i], m=self.m_list[i])

            self.irl_list.append(irl)

        if self.init:
            print("Init Conv")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    print(m)
                    nn.init.xavier_normal_(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        print("No bias here")
            print("End Init Conv")
        else:
            print("Don't Init Conv")
        
        # HopGCN
        self.hopgcn1 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn2 = HopGCN(args, distance_numclass, 64*2, 64)
        self.hopgcn3 = HopGCN(args, distance_numclass, 128*2, 128)
        self.hopgcn4 = HopGCN(args, distance_numclass, 256*2, 256)
        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                                nn.BatchNorm1d(64),
                                                nn.LeakyReLU(negative_slope=0.2))     

    def forward(self, x, p2v_indices, part_rand_idx):
        batch_size, C, N = x.shape
        xyz = x[:, :3, :].clone().detach()  # xyz:(B, 3, N)

        # The kNN graph index used in ISL
        kNN_Graph_idx = knn(xyz, self.k_hat)
        
        part_center, p2v_mask = get_part_feature(
            xyz, p2v_indices, part_rand_idx, part_num=self.args.split_num**3)  # (B, 3, 27)

        # Layer 1
        x1 = self.ISL1(x, kNN_Graph_idx)
        if self.start_layer <= 0:
            x1 = self.irl_list[-4](x1, xyz, kNN_Graph_idx)
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        part_center_embedding = self.conv_pcenteremb(part_center)
        part_feature1 = part_center_embedding + part_feature1  # ()
        g1, hop1_logits = self.hopgcn1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)
        x1 += part2point1
        
        # Layer 2
        x2 = self.ISL2(x1, kNN_Graph_idx)
        if self.start_layer <= 1:
            x2 = self.irl_list[-3](x2, xyz, kNN_Graph_idx)
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g2, hop2_logits = self.hopgcn2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2
        
        
        # Layer 3
        x3 = self.ISL3(x2, kNN_Graph_idx)
        if self.start_layer <= 2:
            x3 = self.irl_list[-2](x3, xyz, kNN_Graph_idx)
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g3, hop3_logits = self.hopgcn3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3

        # Layer 4
        x4 = self.ISL4(x3, kNN_Graph_idx)
        if self.start_layer <= 3:
            x4 = self.irl_list[-1](x4, xyz, kNN_Graph_idx)        
        part_feature4, _ = get_part_feature(
            x4, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)
        g4, hop4_logits = self.hopgcn4(part_feature4)
        part2point4 = get_pointfeature_from_part(g4, p2v_mask, N)  # (B,C,N)
        x4 += part2point4

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn18(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn19(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (hop1_logits, hop2_logits, hop3_logits, hop4_logits)