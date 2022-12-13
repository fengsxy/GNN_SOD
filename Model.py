import torch
import torch.nn as nn
from Module import BatchNorm, GraphSizeNorm,GatedEdgeConv
import torch.nn.functional as F



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
        n_imgs = cnnfeature_1.shape[0]
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

        x1_s = torch.cat((F.relu(self.BN_Re_S_1(self.conv_Re_S_1(x1_s))), cnnfeature_4), dim=1)
        x1_d = torch.cat((F.relu(self.BN_Re_D_1(self.conv_Re_D_1(x1_d))), cnnfeature_4), dim=1)

        x1_s = F.relu(self.BN_S_1(self.conv_S_1(x1_s)))
        x1_d = F.relu(self.BN_D_1(self.conv_D_1(x1_d)))

        x_sdf = x1_s + x1_d

        x_sdf = torch.relu(self.BN1_FDS(self.conv_FDS_1(x_sdf)))

        x2_S = F.relu(self.BN2_S(self.GN(self.conv2_S(x1_S, edge_index_1, edge_attr_1),batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x2_D = F.relu(self.BN2_D(self.GN(self.conv2_D(x1_D, edge_index_2, edge_attr_2),batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)

        x2_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x2_S)
        x2_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x2_D)

        x2_s = torch.stack(x2_s, dim=0).permute(0, 3, 1, 2)
        x2_d = torch.stack(x2_d, dim=0).permute(0, 3, 1, 2)

        x2_s = F.relu(self.BN_Re_S_2(self.conv_Re_S_2(x2_s)))
        x2_d = F.relu(self.BN_Re_D_2(self.conv_Re_D_2(x2_d)))

        x2_s = torch.relu(self.BN_S_2(self.conv_S_2(torch.cat((x2_s, x_sdf, cnnfeature_3), dim=1))))
        x2_d = torch.relu(self.BN_D_2(self.conv_D_2(torch.cat((x2_d, x_sdf, cnnfeature_3), dim=1))))

        x_sdf_2 = torch.relu(self.BN2_FDS(self.conv_FDS_2(x2_s + x2_d)))

        x3_S = F.relu(self.BN3_S(self.GN(self.conv3_S(x2_S, edge_index_1, edge_attr_1),batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x3_D = F.relu(self.BN3_D(self.GN(self.conv3_D(x2_D, edge_index_2, edge_attr_2),batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)

        x3_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x3_S)
        x3_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x3_D)

        x3_s = torch.stack(x3_s, dim=0).permute(0, 3, 1, 2)
        x3_d = torch.stack(x3_d, dim=0).permute(0, 3, 1, 2)

        x3_s = F.relu(self.BN_Re_S_3(self.conv_Re_S_3(x3_s)))
        x3_d = F.relu(self.BN_Re_D_3(self.conv_Re_D_3(x3_d)))

        x3_s = torch.relu(self.BN_S_3(self.conv_S_3(torch.cat((x3_s, x_sdf, cnnfeature_2), dim=1))))
        x3_d = torch.relu(self.BN_D_3(self.conv_D_3(torch.cat((x3_d, x_sdf, cnnfeature_2), dim=1))))

        x_sdf_3 = torch.relu(self.BN_sdf_3(self.conv_sdf_3(x3_s + x3_d)))

        x4_S = F.relu(self.BN4_S(self.GN(self.conv4_S(x3_S, edge_index_1, edge_attr_1),batch_1)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)
        x4_D = F.relu(self.BN4_D(self.GN(self.conv4_D(x3_D, edge_index_2, edge_attr_2),batch_2)))  # + torch.cat((x, torch.zeros(N, 13).cuda()), 1)

        x4_s = self.reverse_map(imgs_num=n_imgs, sp=sp_1, outputs=x4_S)
        x4_d = self.reverse_map(imgs_num=n_imgs, sp=sp_2, outputs=x4_D)

        x4_s = torch.stack(x4_s, dim=0).permute(0, 3, 1, 2)
        x4_d = torch.stack(x4_d, dim=0).permute(0, 3, 1, 2)

        x4_s = F.relu(self.BN_Re_S_4(self.conv_Re_S_4(x4_s)))
        x4_d = F.relu(self.BN_Re_D_4(self.conv_Re_D_4(x4_d)))

        x4_S = torch.relu(self.BN_S_4(self.conv_S_4(torch.cat((x4_s, x_sdf, cnnfeature_1), dim=1))))
        x4_D = torch.relu(self.BN_D_4(self.conv_D_4(torch.cat((x4_d, x_sdf, cnnfeature_1), dim=1))))

        x_sdf_4 = torch.relu(self.BN_sdf_4(self.conv_sdf_4(x4_S + x4_D)))

        x_sdf_out = torch.sigmoid(self.SDF_1_end_conv(x_sdf)).squeeze(1)
        x_sdf_2_out = torch.sigmoid(self.SDF_2_end_conv(x_sdf_2)).squeeze(1)
        x_sdf_3_out = torch.sigmoid(self.SDF_3_end_conv(x_sdf_3)).squeeze(1)
        x_sdf_4_out = torch.sigmoid(self.SDF_4_end_conv(x_sdf_4)).squeeze(1)

        final_out = torch.sigmoid(self.feature_fusion(torch.relu(
            self.bn_all(self.fusion_conv_all(torch.cat([x_sdf, x_sdf_2, x_sdf_3, x_sdf_4], dim=1)))))).squeeze(1)

        return x_sdf_out, x_sdf_2_out, x_sdf_3_out, x_sdf_4_out, final_out
