import torch
import torch.nn as nn
import pytorch_ssim
import pytorch_iou
from Model import  GCN
from torch_geometric.loader import DataLoader
import os, pickle
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from Dataset import MyDataset,PyGDatareader




def PyGDatareader__traintest_merge(path):
    graph_Dataset = []
    path_list = os.listdir(path)
    for i in path_list:
        file_path = os.path.join(path, i)
        train_graphdata = PyGDatareader(file_path)
        graph_Dataset.extend(train_graphdata)
    return graph_Dataset



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