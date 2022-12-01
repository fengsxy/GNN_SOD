import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_ssim
import pytorch_iou
from myEdgeConv import EdgeConv, DynamicEdgeConv
from myGatedEdgeConv import GatedEdgeConv, DynamicGatedEdgeConv, NodeAtt, NodeAtt_wo, Gatedgcn
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.utils import degree, dropout_adj
from torch_geometric.loader import DataLoader
import numpy as np
import os, pickle
import torch.optim as optim
from torch.nn import CrossEntropyLoss
# 0~1 normalization.
from torchvision import transforms
import torchsnooper
from torch.utils.tensorboard import SummaryWriter
from gpu_mem_track import MemTracker


def show_tensor_img(tensor_img: torch.Tensor):
    to_pil = transforms.ToPILImage()
    img = tensor_img.cpu().clone()
    img = to_pil(img)
    img.show()


def get_edge_attr(edge_fusion, target_index_lists):
    result_list = []
    for target_index_list in target_index_lists:
        target_list = []
        for i in target_index_list:
            target_list.append(edge_fusion[i[0]:i[1]])
        y = torch.cat(target_list)
        result_list.append(y)
    return result_list[0], result_list[1], result_list[2]


def generate_index(edge_shape_list):
    generate = np.array([j[1] for i in edge_shape_list for j in i])
    x_list = []
    for i in range(int(len(edge_shape_list))):
        base = generate[:i * 3].sum()
        start_1 = base
        end_1 = base + generate[i * 3]
        start_2 = end_1
        end_2 = start_2 + generate[i * 3 + 1]
        start_3 = end_2
        end_3 = end_2 + generate[i * 3 + 2]
        x_list.append([(start_1, end_1), (start_2, end_2), (start_3, end_3)])
    target = np.array(x_list)
    return target[:, 0, :], target[:, 1, :], target[:, 2, :]


def get_edge_torch(edge_fusion, target_index_lists, edge_skip_list):
    result_list = []
    for num_1, target_index_list in enumerate(target_index_lists, 0):
        target_list = []
        for num_2, i in enumerate(target_index_list, 0):
            if (num_2 == 0):
                target_list.append(edge_fusion[:, i[0]:i[1]])
                print(-edge_skip_list[:num_2 * 3 + num_1].sum() + num_2 * edge_skip_list[num_1])
                print(0)
            else:
                target_list.append(
                    edge_fusion[:, i[0]:i[1]] - edge_skip_list[:num_2 * 3 + 1].sum() + edge_skip_list[num_1])

        y = torch.cat(target_list, axis=1)

        result_list.append(y)
    return result_list[0], result_list[1], result_list[2]


def MaxMinNormalization(x, Max, Min):
    # x = (x-Min)/(Max-Min)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x


def generate_sp_index(edge_shape_list):
    generate = edge_shape_list
    x_list = []
    for i in range(int(len(edge_shape_list) / 3)):
        base = generate[:i * 3].sum()
        start_1 = base
        end_1 = base + generate[i * 3]
        start_2 = end_1
        end_2 = start_2 + generate[i * 3 + 1]
        start_3 = end_2
        end_3 = end_2 + generate[i * 3 + 2]
        x_list.append([(start_1, end_1), (start_2, end_2), (start_3, end_3)])
    target = np.array(x_list)
    return target[:, 0, :], target[:, 1, :], target[:, 2, :]


def get_sp_feature(edge_fusion, target_index_lists):
    result_list = []
    for target_index_list in target_index_lists:
        target_list = []
        for i in target_index_list:
            target_list.append(edge_fusion[i[0]:i[1], :])
        y = torch.cat(target_list)
        result_list.append(y)
    return result_list[0], result_list[1], result_list[2]


def get_batch(batch_fusion, batch_index_lists):
    result_list = []
    for target_index_list in batch_index_lists:
        target_list = []
        for i in target_index_list:
            target_list.append(batch_fusion[i[0]:i[1]])
        y = torch.cat(target_list)
        result_list.append(y)
    return result_list[0], result_list[1], result_list[2]


def PyGDatareader__traintest_merge(path):
    graph_Dataset = []
    path_list = os.listdir(path)
    for i in path_list:
        file_path = os.path.join(path, i)
        train_graphdata = PyGDatareader(file_path)
        graph_Dataset.extend(train_graphdata)
    return graph_Dataset


class BatchNorm(torch.nn.BatchNorm1d):
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        """"""
        return super(BatchNorm, self).forward(x)

    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)


class GraphSizeNorm(nn.Module):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """

    def __init__(self):
        super(GraphSizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1)


class GCN(nn.Module):
    def __init__(self, in_features=4, out_features=1):
        super(GCN, self).__init__()
        self.mean5 = torch.Tensor([[122.67892, 116.66877, 104.00699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / 255.0
        self.mean4 = torch.Tensor([[122.67892, 116.66877, 104.00699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / 255.0
        self.mean3 = torch.Tensor([[122.67892, 116.66877, 104.00699, 0, 0, 0, 0, 0, 0, 0, 0]]) / 255.0
        self.mean2 = torch.Tensor([[122.67892, 116.66877, 104.00699, 0, 0, 0, 0, 0, 0]]) / 255.0
        self.mean1 = torch.Tensor([[0, 0, 122.67892, 116.66877, 104.00699, 0, 0, 0, 0]]) / 255.0
        self.GN = GraphSizeNorm()

        self.conv1_F = GatedEdgeConv(1536, 64)
        self.BN1_F = BatchNorm(in_channels=64)

        self.conv2_F = GatedEdgeConv(64, 32)
        self.BN2_F = BatchNorm(in_channels=32)

        self.conv3_F = GatedEdgeConv(32, 16)
        self.BN3_F = BatchNorm(in_channels=16)

        self.conv4_F = GatedEdgeConv(16, 8)
        self.BN4_F = BatchNorm(in_channels=8)

        self.conv1_D = GatedEdgeConv(1536, 64)
        self.BN1_D = BatchNorm(in_channels=64)

        self.conv2_D = GatedEdgeConv(64, 32)
        self.BN2_D = BatchNorm(in_channels=32)

        self.conv3_D = GatedEdgeConv(32, 16)
        self.BN3_D = BatchNorm(in_channels=16)

        self.conv4_D = GatedEdgeConv(16, 8)
        self.BN4_D = BatchNorm(in_channels=8)

        # for p in self.parameters():
        #     p.requires_grad = False
        # # 首先固定运动网络，只训练静态网络
        #############################################################
        # static net
        K = 6

        self.conv1_S = GatedEdgeConv(1536, 64)
        self.BN1_S = BatchNorm(in_channels=64)

        self.conv2_S = GatedEdgeConv(64, 32)
        self.BN2_S = BatchNorm(in_channels=32)

        self.conv3_S = GatedEdgeConv(32, 16)
        self.BN3_S = BatchNorm(in_channels=16)

        self.conv4_S = GatedEdgeConv(16, 8)
        self.BN4_S = BatchNorm(in_channels=8)

        self.conv_Re_F_1 = nn.Conv2d(64, 64, (1, 1))
        self.conv_Re_D_1 = nn.Conv2d(64, 64, (1, 1))
        self.conv_Re_S_1 = nn.Conv2d(64, 64, (1, 1))
        self.BN_Re_F_1 = nn.BatchNorm2d(num_features=64)
        self.BN_Re_D_1 = nn.BatchNorm2d(num_features=64)
        self.BN_Re_S_1 = nn.BatchNorm2d(num_features=64)

        self.conv_F_1 = nn.Conv2d(128, 64, (1, 1))
        self.conv_D_1 = nn.Conv2d(128, 64, (1, 1))
        self.conv_S_1 = nn.Conv2d(128, 64, (1, 1))
        self.BN_F_1 = nn.BatchNorm2d(num_features=64)
        self.BN_D_1 = nn.BatchNorm2d(num_features=64)
        self.BN_S_1 = nn.BatchNorm2d(num_features=64)
        self.conv_FDS_1 = nn.Conv2d(64, 32, (1, 1))

        self.BN1_FDS = nn.BatchNorm2d(num_features=32)
        self.BN2_FDS = nn.BatchNorm2d(num_features=16)
        self.conv_FDS_2 = nn.Conv2d(32, 16, (1, 1))
        self.conv_Re_F_2 = nn.Conv2d(32, 32, (1, 1))
        self.conv_Re_D_2 = nn.Conv2d(32, 32, (1, 1))
        self.conv_Re_S_2 = nn.Conv2d(32, 32, (1, 1))
        self.BN_Re_F_2 = nn.BatchNorm2d(num_features=32)
        self.BN_Re_D_2 = nn.BatchNorm2d(num_features=32)
        self.BN_Re_S_2 = nn.BatchNorm2d(num_features=32)

        self.conv_F_2 = nn.Conv2d(96, 32, (1, 1))
        self.conv_D_2 = nn.Conv2d(96, 32, (1, 1))
        self.conv_S_2 = nn.Conv2d(96, 32, (1, 1))
        self.BN_F_2 = nn.BatchNorm2d(num_features=32)
        self.BN_D_2 = nn.BatchNorm2d(num_features=32)
        self.BN_S_2 = nn.BatchNorm2d(num_features=32)

        self.conv_Re_F_3 = nn.Conv2d(16, 16, (1, 1))
        self.conv_Re_D_3 = nn.Conv2d(16, 16, (1, 1))
        self.conv_Re_S_3 = nn.Conv2d(16, 16, (1, 1))
        self.BN_Re_F_3 = nn.BatchNorm2d(num_features=16)
        self.BN_Re_D_3 = nn.BatchNorm2d(num_features=16)
        self.BN_Re_S_3 = nn.BatchNorm2d(num_features=16)
        self.conv_F_3 = nn.Conv2d(64, 16, (1, 1))
        self.conv_D_3 = nn.Conv2d(64, 16, (1, 1))
        self.conv_S_3 = nn.Conv2d(64, 16, (1, 1))
        self.BN_F_3 = nn.BatchNorm2d(num_features=16)
        self.BN_D_3 = nn.BatchNorm2d(num_features=16)
        self.BN_S_3 = nn.BatchNorm2d(num_features=16)

        self.conv_sdf_3 = nn.Conv2d(16, 8, (1, 1))
        self.BN_sdf_3 = nn.BatchNorm2d(num_features=8)
        self.conv_sdf_4 = nn.Conv2d(8, 4, (1, 1))
        self.BN_sdf_4 = nn.BatchNorm2d(num_features=4)

        self.conv_Re_F_4 = nn.Conv2d(8, 8, (1, 1))
        self.conv_Re_D_4 = nn.Conv2d(8, 8, (1, 1))
        self.conv_Re_S_4 = nn.Conv2d(8, 8, (1, 1))
        self.BN_Re_F_4 = nn.BatchNorm2d(num_features=8)
        self.BN_Re_D_4 = nn.BatchNorm2d(num_features=8)
        self.BN_Re_S_4 = nn.BatchNorm2d(num_features=8)
        self.conv_F_4 = nn.Conv2d(48, 8, (1, 1))
        self.conv_D_4 = nn.Conv2d(48, 8, (1, 1))
        self.conv_S_4 = nn.Conv2d(48, 8, (1, 1))
        self.BN_F_4 = nn.BatchNorm2d(num_features=8)
        self.BN_D_4 = nn.BatchNorm2d(num_features=8)
        self.BN_S_4 = nn.BatchNorm2d(num_features=8)

        self.SDF_1_end_conv = nn.Conv2d(32, 1, (1, 1))
        self.SDF_2_end_conv = nn.Conv2d(16, 1, (1, 1))
        self.SDF_3_end_conv = nn.Conv2d(8, 1, (1, 1))
        self.SDF_4_end_conv = nn.Conv2d(4, 1, (1, 1))

        self.Up_1 = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True).cuda()
        self.feature_4_conv = nn.Conv2d(512, 64, (1, 1))
        self.feature_3_conv = nn.Conv2d(512, 32, (1, 1))
        self.feature_2_conv = nn.Conv2d(256, 16, (1, 1))
        self.feature_1_conv = nn.Conv2d(128, 8, (1, 1))

        self.fusion_conv_all = nn.Conv2d(60, 60, (1, 1))
        self.bn_all = nn.BatchNorm2d(60)
        self.feature_fusion = nn.Conv2d(60, 1, (1, 1))

    def reverse_map(self, imgs_num, outputs, sp):
        salmapList = []
        ind = 0
        for i in range(0, imgs_num):
            superpixel = sp[i].type(torch.int64)
            superpixelCPU = superpixel
            sp_indices = torch.unique(superpixelCPU)
            n_sp_extracted = len(sp_indices)
            # make sure superpixel indices are numbers from 0 to n-1
            assert n_sp_extracted == superpixelCPU.max() + 1, ('superpixel indices', torch.unique(superpixelCPU))
            salmapList.append(outputs[ind:ind + n_sp_extracted][superpixel])
            ind = ind + n_sp_extracted
        return salmapList

    def sp_mp_sub(self, superpixels, sp, avg_values, feature):
        mask = superpixels == sp
        avg_values[sp, :] = torch.mean(feature[:, mask])

    # def sp_mp_multi_process(self,feature, superpixels):
    #    n_sp = len(torch.unique(superpixels))
    #    n_ch = feature.shape[1]
    #    n_batch = feature.shape[0]
    #    avg_list = []
    #    avg_values = torch.zeros((n_sp,n_batch, n_ch)).cuda(0)  # 创建数组，用于存储每个超像素的特征平均值，分通道存储。
    #    feature = feature.permute(1,0,2,3)
    #    import torch.multiprocessing as mp
    #    mp.Process(target=self.sp_mp_sub,args=(superpixels,n_sp,avg_vaules,feature))

    def sp_map(self, feature, superpixels):

        n_sp = len(torch.unique(superpixels))
        n_ch = feature.shape[1]
        n_batch = feature.shape[0]
        avg_list = []
        avg_values = torch.zeros((n_sp, n_batch, n_ch)).cuda(0)  # 创建数组，用于存储每个超像素的特征平均值，分通道存储。
        feature = feature.permute(1, 0, 2, 3)
        for batch_num in range(n_batch):
            for sp in torch.unique(superpixels.cuda(0)):
                mask = superpixels[batch_num] == sp  # 对array做逻辑判断，则返回一个等大的数组，superpixel中元素值与sp相等的位置为true，不等的位置为false
                avg_values[sp, batch_num, :] = torch.mean(
                    feature[:, :, mask])  # img[:, :, c]是指定图像第c通道的矩阵值，然后利用[mask]进行索引取出值进行求平均。
        return avg_values

    def forward(self, data_list, batch_list):
        # sp_feature的前两列是coord，后三列是RGB三通道平均值
        sp_feature_1, edge_index_1, edge_attr_1 = data_list[0].x, data_list[0].edge_index, data_list[0].edge_attr
        sp_feature_2, edge_index_2, edge_attr_2 = data_list[1].x, data_list[1].edge_index, data_list[1].edge_attr
        sp_feature_3, edge_index_3, edge_attr_3 = data_list[2].x, data_list[2].edge_index, data_list[2].edge_attr
        batch_1, batch_2, batch_3 = batch_list[0], batch_list[1], batch_list[2]
        sp_1 = data_list[0].sp
        sp_2 = data_list[1].sp
        sp_3 = data_list[2].sp
        cnnfeature_list = data_list[0].cnnfeatureList
        feature_1 = torch.stack([torch.tensor(i[0]).cuda() for i in cnnfeature_list], dim=0)
        feature_2 = torch.stack([torch.tensor(i[1]).cuda() for i in cnnfeature_list], dim=0)
        feature_3 = torch.stack([torch.tensor(i[2]).cuda() for i in cnnfeature_list], dim=0)
        feature_4 = torch.stack([torch.tensor(i[3]).cuda() for i in cnnfeature_list], dim=0)

        cnnfeature_4 = self.feature_4_conv(self.Up_1(feature_1))
        cnnfeature_3 = self.feature_3_conv(self.Up_1(feature_2))
        cnnfeature_2 = self.feature_2_conv(self.Up_1(feature_3))
        cnnfeature_1 = self.feature_1_conv(self.Up_1(feature_4))
        n_imgs =cnnfeature_1.shape[0]
        sp_feature_1[:, 0:9] = sp_feature_1[:, 0:9] - self.mean1.cuda(0)
        x_S = sp_feature_1[:, 5:]

        x1_S = F.relu(self.BN1_S(
            self.GN(self.conv1_S(x_S, edge_index_1, edge_attr_1),
                    batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)

        sp_feature_2[:, 0:9] = sp_feature_2[:, 0:9] - self.mean1.cuda(0)
        x_D = sp_feature_2[:, 5:]
        x1_D = F.relu(self.BN1_D(
            self.GN(self.conv1_D(x_D, edge_index_2, edge_attr_2),
                    batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)

        # X1_S,X1_D,X1_F
        x1_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x1_S)
        x1_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x1_D)


        x1_s = torch.stack(x1_s, dim=0).permute(0, 3, 1, 2)
        x1_d = torch.stack(x1_d, dim=0).permute(0, 3, 1, 2)


        x1_s = torch.cat((F.relu(self.BN_Re_S_1(self.conv_Re_S_1(x1_s))) ,cnnfeature_4),dim=1)
        x1_d = torch.cat((F.relu(self.BN_Re_D_1(self.conv_Re_D_1(x1_d))),cnnfeature_4),dim=1)


        x1_s = F.relu(self.BN_S_1(self.conv_S_1(x1_s)))
        x1_d = F.relu(self.BN_D_1(self.conv_D_1(x1_d)))

        x_sdf = x1_s + x1_d

        x_sdf = torch.relu(self.BN1_FDS(self.conv_FDS_1(x_sdf)))

        x2_S = F.relu(self.BN2_S(self.GN(self.conv2_S(x1_S, edge_index_1, edge_attr_1),
                                         batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x2_D = F.relu(self.BN2_D(self.GN(self.conv2_D(x1_D, edge_index_2, edge_attr_2),
                                         batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)


        x2_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x2_S)
        x2_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x2_D)

        x2_s = torch.stack(x2_s, dim=0).permute(0, 3, 1, 2)
        x2_d = torch.stack(x2_d, dim=0).permute(0, 3, 1, 2)

        x2_s = F.relu(self.BN_Re_S_2(self.conv_Re_S_2(x2_s)))
        x2_d = F.relu(self.BN_Re_D_2(self.conv_Re_D_2(x2_d)))

        x2_s = torch.relu(self.BN_S_2(self.conv_S_2(torch.cat((x2_s , x_sdf ,cnnfeature_3),dim=1))))
        x2_d = torch.relu(self.BN_D_2(self.conv_D_2(torch.cat((x2_d , x_sdf ,cnnfeature_3),dim=1))))


        x_sdf_2 = torch.relu(self.BN2_FDS(self.conv_FDS_2(x2_s + x2_d )))

        x3_S = F.relu(self.BN3_S(self.GN(self.conv3_S(x2_S, edge_index_1, edge_attr_1),
                                         batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x3_D = F.relu(self.BN3_D(self.GN(self.conv3_D(x2_D, edge_index_2, edge_attr_2),
                                         batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)


        x3_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x3_S)
        x3_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x3_D)

        x3_s = torch.stack(x3_s, dim=0).permute(0, 3, 1, 2)
        x3_d = torch.stack(x3_d, dim=0).permute(0, 3, 1, 2)

        x3_s = F.relu(self.BN_Re_S_3(self.conv_Re_S_3(x3_s)))
        x3_d = F.relu(self.BN_Re_D_3(self.conv_Re_D_3(x3_d)))

        x3_s = torch.relu(self.BN_S_3(self.conv_S_3(torch.cat((x3_s , x_sdf ,cnnfeature_2),dim=1))))
        x3_d = torch.relu(self.BN_D_3(self.conv_D_3(torch.cat((x3_d , x_sdf ,cnnfeature_2),dim=1))))

        x_sdf_3 = torch.relu(self.BN_sdf_3(self.conv_sdf_3(x3_s  + x3_d)))

        x4_S = F.relu(self.BN4_S(self.GN(self.conv4_S(x3_S, edge_index_1, edge_attr_1),
                                         batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x4_D = F.relu(self.BN4_D(self.GN(self.conv4_D(x3_D, edge_index_2, edge_attr_2),
                                         batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)


        x4_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x4_S)
        x4_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x4_D)


        x4_s = torch.stack(x4_s, dim=0).permute(0, 3, 1, 2)
        x4_d = torch.stack(x4_d, dim=0).permute(0, 3, 1, 2)


        x4_s = F.relu(self.BN_Re_S_4(self.conv_Re_S_4(x4_s)))
        x4_d = F.relu(self.BN_Re_D_4(self.conv_Re_D_4(x4_d)))


        x4_S = torch.relu(self.BN_S_4(self.conv_S_4(torch.cat((x4_s , x_sdf ,cnnfeature_1),dim=1))))
        x4_D = torch.relu(self.BN_D_4(self.conv_D_4(torch.cat((x4_d , x_sdf ,cnnfeature_1),dim=1))))

        x_sdf_4 = torch.relu(self.BN_sdf_4(self.conv_sdf_4(x4_S +  x4_D)))

        x_sdf_out = torch.sigmoid(self.SDF_1_end_conv(x_sdf)).squeeze(1)
        x_sdf_2_out = torch.sigmoid(self.SDF_2_end_conv(x_sdf_2)).squeeze(1)
        x_sdf_3_out = torch.sigmoid(self.SDF_3_end_conv(x_sdf_3)).squeeze(1)
        x_sdf_4_out = torch.sigmoid(self.SDF_4_end_conv(x_sdf_4)).squeeze(1)

        final_out = torch.sigmoid(self.feature_fusion(torch.relu(
            self.bn_all(self.fusion_conv_all(torch.cat([x_sdf, x_sdf_2, x_sdf_3, x_sdf_4], dim=1)))))).squeeze(1)

        return x_sdf_out, x_sdf_2_out, x_sdf_3_out, x_sdf_4_out, final_out


class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, path):
        self.path_list = os.listdir(path)
        self.path_list = [os.path.join(path, i) for i in self.path_list]
        self.len = len(self.path_list)

    def __getitem__(self, index):
        return PyGDatareader(self.path_list[index])

    def __len__(self):
        return self.len


def PyGDatareader(datafile_path):
    graphdata_List = []
    assert os.path.isfile(datafile_path), print('Invalid datafile_path {%s}' % datafile_path)
    # print('loading graph data from %s' % datafile_path)
    with open(datafile_path, 'rb') as f:
        while True:
            try:
                graphdata = pickle.load(f)
                graphdata_List.extend(graphdata)
            except EOFError:
                break
    return graphdata_List


bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def muti_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    path = "./TXT/data_mini_processed"
    trainset = MyDataset(path)  # Combines a train_data

    batch_size = 1

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GCN().to(device)

    #model.load_state_dict(torch.load("./model/model_09_Finall_test_30.pkl"))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)
    epoch_num = 20
    max = 999999999.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        training_loss = 0.0
        loss = 0
        flag = 1
        img_name_list = []
        txt_path = "./TXT/EORSSD_test.txt"
        fh = open(txt_path, 'r')
        for line in fh:
            line = line.rstrip()  # remove the "space" in the end of the string line.
            words = line.split()  # split the line into serval string with "space".
            # file_name = words[1].split('/')[-2]  # word[0] is the image path, word[1] is the gt path
            file_name = words[1].split('/')[-2]
            img_name = words[1].split('/')[-1]
            img_name_list.append(file_name + '/' + img_name)
        for batch_idx, gt_graph_sp_PyGdata in enumerate(trainloader, 0):
            gt_graph_sp_PyGdata_1, gt_graph_sp_PyGdata_2, gt_graph_sp_PyGdata_3 = gt_graph_sp_PyGdata[0]
            gt = gt_graph_sp_PyGdata_1.y  # get the batch groundtruth labels (tensor: B*H*W)

            n_imgs = gt.shape[0]
            gt_graph_sp_PyGdata_1 = gt_graph_sp_PyGdata_1.to(device)
            gt_graph_sp_PyGdata_2 = gt_graph_sp_PyGdata_2.to(device)
            gt_graph_sp_PyGdata_3 = gt_graph_sp_PyGdata_3.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()  # clear the grad in the optimizer.
            # de-batch the outputs
            batch_1 = gt_graph_sp_PyGdata_1.batch
            batch_2 = gt_graph_sp_PyGdata_2.batch
            batch_3 = gt_graph_sp_PyGdata_3.batch
            #############################################################################2######
            ##forward

            x_sdf_out, x_sdf_2_out, x_sdf_3_out, x_sdf_4_out, salmaps = model(
                data_list=(gt_graph_sp_PyGdata_1, gt_graph_sp_PyGdata_2, gt_graph_sp_PyGdata_3),
                batch_list=(batch_1, batch_2, batch_3))  # return a tensor BN*C

            gtlist = []
            for i in range(0, n_imgs):
                gtlist.append(gt[i].squeeze(0))
            gts = torch.cat(gtlist, dim=0)

            gts = gts.reshape(int(gts.shape[0] / gts.shape[1]), gts.shape[1], gts.shape[1])

            loss_fn = CrossEntropyLoss()  # self-defined loss function, call loss to compute.
            loss_1 = muti_loss(salmaps, gt.cuda(0))
            loss_2 = muti_loss(x_sdf_out, gt.cuda(0))
            loss_3 = muti_loss(x_sdf_2_out, gt.cuda(0))
            loss_4 = muti_loss(x_sdf_3_out, gt.cuda(0))
            loss_5 = muti_loss(x_sdf_4_out, gt.cuda(0))
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

            loss.backward()  # compute the grad.

            training_loss += loss.item()
            optimizer.step()  # update the weight using the grad.
            # print('aa is ', aa)
        if max > training_loss:
            torch.save(model.state_dict(), "./model/model_09_100_300_cat" + '_' + '%d.pkl' % (epoch))
            max = training_loss
