#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Construct the spatial graph data using superpixels for the given training/testing dataset.
# The graph data is consisted of adjacent matrix or weight matrix, and node features, node coord.
import torch
import numpy as np

# from matplotlib import pyplot as plt
import scipy.ndimage
from scipy.spatial.distance import cdist
import argparse
import datetime
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import multiprocessing as mp
import pickle
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import dense_to_sparse
import skimage.color
from pysnic.algorithms.snic import snic
import os

############################################################################


transform = transforms.Compose([transforms.ToTensor()])


##Arguments parser described with "Extract SLIC superpixels from image and construct the spatial graph".
def parse_args(param):
    # 创建一个解析对象
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from image and construct the spatial graph')
    # 向该对象中添加要关注的参数和选项
    parser.add_argument('-D', '--dataset', type=str, default='EORSSD', choices=['DAVIS', 'FBMS', 'MCL', 'SegV2',
                                                                                'UVSD', 'ViSal', 'VOS', 'DAVSOD',
                                                                                'DUTS', 'ECSSD', 'HKU-IS', 'MSRA-10k',
                                                                                'MSRA-10k-2', 'MSRA-10k-3',
                                                                                'MSRA-10k-4',
                                                                                'PASCAL-S', 'EORSSD'])
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to the dataset')
    parser.add_argument('-o', '--out_dir', type=str, default='./data_test_processed',
                        help='path where to save graph and sp')
    parser.add_argument('-s', '--split', type=str, default='train',
                        choices=['train', 'train_1', 'train_2', 'val', 'test', 'train_small', 'test_small',
                                 'train_test'])
    parser.add_argument('-t', '--threads', type=int, default=-1, help='number of parallel threads')
    parser.add_argument('-n', '--n_sp_query', type=int, default=500, help='max number of superpixels per image')
    parser.add_argument('-k', '--knn_graph', type=int, default=32, help='maximum number of neighbors for each node')
    #########################################
    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=True)
    # Testing settings
    parser.add_argument('--model', type=str, default='./epoch_resnet.pth')
    # parser.add_argument('--test_fold', type=str, default='./results/test')
    # parser.add_argument('--test_mode', type=int, default=1)
    # parser.add_argument('--sal_mode', type=str, default='t')
    # Misc
    # parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    #########################################
    # 进行解析
    argus = parser.parse_args(param)
    return argus


# sparse the spatial adjacent (weight) matrix, only preserve the knn neighbors in eulic distance space for each sp node.
def sparsify_graph(A, knn_graph):
    if knn_graph is not None and knn_graph < A.shape[0]:  # 在Python 中一切都是对象，毫无例外整数也是对象，对象之间比较是否相等可以用==，也可以用is
        # not可以作为逻辑运算符使用，也可以作为表达式的一部分，此处是作为表达式用
        # Python中的None，是一个特殊的常量，空值，是Python里的一个特殊的值。
        idx = np.argsort(A, axis=0)[:-knn_graph, :]  # 取邻接矩阵A中每列除最大的knn_graph个元素之外的所有元素在原矩阵A中的索引值，并在后续将索引值对应元素置为0；
        # argsort(A)函数首先将数组A的值从小到大排序后，然后按照其相对应的索引值输出；
        # argsort(A,axis=0)将数组A按列排序（对每列分别从小到大排列），然后取每列排序后的索引值组成新数组；
        # argsort(A,axis=1)将数组A按行排序（对每行分别从小到大排列），然后取每行排序后的索引值组成新数组；
        # np.argsort()[num]：就是对提取出来的索引值数组进行[num]索引；
        # 当num>=0时，np.argsort()[num]就可以理解为y[num];
        # 当num<0时，np.argsort()[num]就是把输出的索引数组的元素反向输出；
        # put_along_axis(arr, indices, values, axis)
        # Put values into the destination array by matching 1d index and data slices.
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]  # 同上，取邻接矩阵A中每行除最大的knn_graph个元素之外的所有元素在原矩阵A中的索引值，并在后续将索引值对应元素置为0；
        np.put_along_axis(A, idx, 0, axis=1)  # 将上面取到的索引值对应的元素置为0
    return A


# generate the spatial adjacent matrix (weight matrix) in coord eucli-space
def spatial_graph(coord, img_size, knn_graph=32):
    coord = coord / np.array(img_size, np.float32)  # 用图像的height，width来对所有超像素中心坐标值归一化。#float:64bit；float32:32bit.
    dist = cdist(coord, coord)  # 计算两两超像素间的欧式空间距离，生成n_sp*n_sp的距离方阵
    # b=max(dist())
    # a=max(max(dist))
    # dist = dist / max(max(dist))
    sigma = 0.1 * np.pi  # 3.1415926*0.1
    A = np.exp(- dist / sigma ** 2)  # 距离方阵逐元素高斯化，生成稠密的邻接矩阵（本身是权值矩阵）
    A[np.diag_indices_from(A)] = 0  # remove self-loops，对角线元素置为0
    sparsify_graph(A, knn_graph)  # 对距离方阵稀疏化，即距离远到一定程度，即A元素的值小到一定程度就置为0
    return A  # adjacency matrix (edges)


# visualize the superpixels with the avg value in one sp in the original image
def visualize_superpixels(avg_values, superpixels):
    n_ch = avg_values.shape[1]  # the num of channel
    img_sp = np.zeros((*superpixels.shape, n_ch))  # 创建一个大小为 height*width*channel的全零矩阵img_sp
    # *号表示可变参数，函数的可变参数是指传入的参数是可以变化的，1个，2个到任意个
    for sp in np.unique(superpixels):
        mask = superpixels == sp  # 对array做逻辑判断，则返回一个等大的数组，superpixel中元素值与sp相等的位置为true，不等的位置为false
        for c in range(n_ch):  # 对相应超像素区域，分通道进行超像素平均特征值赋值
            img_sp[:, :, c][mask] = avg_values[sp, c]
    return img_sp


# Compute the avg color of the sp(N*3), center coord of the sp (N*2), and the list of mask (index of pixels in all sp)
def superpixel_features(img, superpixels):
    # RGB 3Chanels is
    n_sp = len(np.unique(superpixels))  # Count the num of the sp in fact
    # numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
    # Find the unique elements of an array, and return the sorted unique elements of an array(in a ascenging order)
    # param: ar:array-like: input array. Unless axis is specified, this will be flattened if it is not already 1-D.
    n_ch = img.shape[2]  # 对于输入的图像，维度是height*width*channel，此处获取输入图像的通道数
    avg_values = np.zeros((n_sp, n_ch))  # 创建数组，用于存储每个超像素的特征平均值，分通道存储。
    coord = np.zeros((n_sp, 2))  # 创建数组，用于存储每个超像素的坐标值。
    masks = []
    # 循环遍历所有超像素，计算每个超像素的特征平均值（分channel），坐标中心，超像素内所有像素点的索引
    for sp in np.unique(superpixels):
        mask = superpixels == sp  # 对array做逻辑判断，则返回一个等大的数组，superpixel中元素值与sp相等的位置为true，不等的位置为false。
        for c in range(n_ch):  # 对每个通道分别进行处理
            avg_values[sp, c] = np.mean(img[:, :, c][mask])  # img[:, :, c]是指定图像第c通道的矩阵值，然后利用[mask]进行索引取出值进行求平均。
        coord[sp] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        # scipy.ndimage.measurements.center_of_mass(input, labels=None, index=None)
        # Calculate the center of mass of the values of an array at labels.
        # param,input:ndarray,Data from which to calculate center-of-mass.
        masks.append(mask)  # append()函数
        # 描述：在列表ls最后(末尾)添加一个元素object
        # 语法：ls.append(object) -> None 无返回值
        # object —— 要添加的元素。可以添加 列表，字典，元组，集合，字符串等。
        # avg_values [segment_max_num,channel] coord[x,y] mask列表
    return avg_values, coord, masks


# Compute the avg color of the sp(N*3), center coord of the sp (N*2), and the list of mask (index of pixels in all sp)
def superpixel_features_CNN(imgRGB, superpixels, Model):
    # cnnfeature = CNNpspnet5layer.CNNfeature(img=img, CNN_model=CNNmodel)
    # mobile = Fastnet()
    # # define a CNN model
    # Model = Solver(None, None, config=args, save_fold=None)
    # transform = transforms.Compose([transforms.ToTensor()])
    # in_ -= np.array((104.00699, 116.66877, 122.67892))
    # img_ = np.array(img, dtype=np.float32) * 255.0
    # 注意，这里的img是RGB格式，而这个CNN网络要求输入BGR格式，因此进行反序操作；且需要去掉通道均值，并放大到0~255范围。
    imgBGR = imgRGB[:, :, ::-1].copy()
    imgBGR -= np.array((104.00699, 116.66877, 122.67892)) / 255.0
    imgT01 = torch.Tensor(imgBGR).unsqueeze(0).permute(0, 3, 1, 2)  # B,C,H,W
    imgT255 = imgT01 * 255.0
    # cnnfeature = mobile(imgT)
    cnnfeatureList = Model.test(imgs=imgT255)
    cnnfeature = torch.cat((torch.sigmoid(cnnfeatureList[0]), torch.sigmoid(cnnfeatureList[1]),
                            torch.sigmoid(cnnfeatureList[2]), torch.sigmoid(cnnfeatureList[3])), dim=1).permute(0, 2, 3,
                                                                                                                1).cpu().data.numpy().squeeze(
        0)
    n_sp = len(np.unique(superpixels))  # Count the num of the sp in fact
    # numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
    # Find the unique elements of an array, and return the sorted unique elements of an array(in a ascenging order)
    # param: ar:array-like: input array. Unless axis is specified, this will be flattened if it is not already 1-D.
    # n_ch = img.shape[2]  # 对于输入的图像，维度是height*width*channel，此处获取输入图像的通道数
    n_ch = cnnfeature.shape[2]
    avg_values = np.zeros((n_sp, n_ch))  # 创建数组，用于存储每个超像素的特征平均值，分通道存储。
    # coord = np.zeros((n_sp, 2))  # 创建数组，用于存储每个超像素的坐标值。
    # masks = []
    # 循环遍历所有超像素，计算每个超像素的特征平均值（分channel），坐标中心，超像素内所有像素点的索引
    for sp in np.unique(superpixels):
        mask = superpixels == sp  # 对array做逻辑判断，则返回一个等大的数组，superpixel中元素值与sp相等的位置为true，不等的位置为false。
        for c in range(n_ch):  # 对每个通道分别进行处理
            avg_values[sp, c] = np.mean(cnnfeature[:, :, c][mask])  # img[:, :, c]是指定图像第c通道的矩阵值，然后利用[mask]进行索引取出值进行求平均。
        # coord[sp] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        # scipy.ndimage.measurements.center_of_mass(input, labels=None, index=None)
        # Calculate the center of mass of the values of an array at labels.
        # param,input:ndarray,Data from which to calculate center-of-mass.
        # masks.append(mask)  # append()函数
        # 描述：在列表ls最后(末尾)添加一个元素object
        # 语法：ls.append(object) -> None 无返回值
        # object —— 要添加的元素。可以添加 列表，字典，元组，集合，字符串等。
    return avg_values  # , coord, masks


##Compute the super-pixels for the given image and generate the spatial graph;
##return the spatial weight matrix (A_spatial), node features (sp_intensity, sp_coord) and superpixels.
def process_image(params):
    img, index, n_images, args, to_print, label, img_size, Model = params
    # img, index, n_images, args, to_print, label, img_size = params
    n_sp_query_list = [100, 300, 500]
    knn_graph = args.knn_graph
    # #############################SLIC############################

    # extract the superpixels and compute the features of superpixels
    # superpixels = slic(img, n_segments=n_sp_query, compactness=5.0)  # img: 2D, 3D or 4D ndarray, element is double type required.
    # # #############################SNIC############################
    lab_image = skimage.color.rgb2lab(img).tolist()
    label = torch.from_numpy(label).permute(2, 0, 1)
    graph_data_list = []
    for n_sp_query in n_sp_query_list:
        segmentation, _, number_of_segments = snic(lab_image, n_sp_query, 1)
        superpixels = np.array(segmentation)
        ###############################################################
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))
        # make sure superpixel indices are numbers from 0 to n-1
        sp_intensity_RGB, sp_coord, masks = superpixel_features(img, superpixels)
        sp_intensity_CNN = superpixel_features_CNN(img, superpixels, Model)
        sp_intensity = np.concatenate((sp_intensity_RGB, sp_intensity_CNN), axis=1)

        # construct the spatial graph in the coord euclidean-space, return a N*N adjacent matrix
        A_spatial = spatial_graph(sp_coord, img.shape[:2], knn_graph=knn_graph)
        # Sparse the adjacent matrix
        A_spatial = torch.from_numpy(A_spatial)  # firstly convert ndarray to tensor
        # A_sparse = A_spatial.to_sparse()  # sparse the dense matrix
        edge_index, edge_attr = dense_to_sparse(A_spatial)
        sp_feature = torch.from_numpy(
            np.concatenate((sp_coord, sp_intensity), axis=1)).float()  # numpy N*C → tensor N*C

        superpixels = torch.from_numpy(superpixels).unsqueeze(0)
        # include adjacent mat, node features, gt, superpixel for the given image.
        edge_attr = edge_attr.type(sp_feature.dtype)  # make sure that edge_attr has the same type with x.

        graphdata = PyGData(x=sp_feature, edge_index=edge_index,
                            edge_attr=edge_attr, y=label, sp=superpixels,
                            img_size=img_size)
        graph_data_list.append(graphdata)
    return graph_data_list


# Define our specified dataset class
# reference: https://blog.csdn.net/u011995719/article/details/85102770
class MyDataset(Dataset):
    # __init__() extract the content in txt file to a list saved in img_gt_List.
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        img_gt_List = []
        for line in fh:
            line = line.rstrip()  # remove the "space" in the end of the string line.
            words = line.split()  # split the line into serval string with "space".
            img_gt_List.append((words[0], words[1]))  # word[0] is the image path, word[1] is the gt path
        self.img_gt_List = img_gt_List
        self.transform = transform
        self.target_transform = target_transform

    # read the data and label according to the given index
    def __getitem__(self, index):
        fn, label = self.img_gt_List[index]
        # PIL.Image读取彩色图片：RGB， size:(w,h)，通道值范围0-255；转成numpy后变成(h,w,c)；转换成tensor后变为（c,h,w）
        img = Image.open(fn).convert('RGB')  # input data is converted to RGB color image,
        gt = Image.open(label).convert('L')  # input gt is converted to L (grayscale) image
        ##################################
        # resize the image and gt to the fixed size 300*300  in transform
        ##################################
        # transforms.ToTensor()
        shape_tuple = gt.size  # save the origianl image size.
        shape_np = np.zeros((1, 2))
        shape_np[0:2] = [shape_tuple[0], shape_tuple[1]]
        img_size = torch.from_numpy(shape_np)
        if self.transform is not None:  # convert the PIL.Image object to a tensor.
            img = self.transform(img)  # resize image to a specific size, so it's easy to process in batch.
            gt = self.transform(gt)
        return img, gt, img_size  # return tensors, c*h*w, range 0~255; original img size, h*w.

    # len the datasets (the num of the data samples)
    def __len__(self):
        return len(self.img_gt_List)


# Generate the spatial graph data for the given dataset.
if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    # mp.set_start_method('spawn')  # multiple threads of GPU

    #######################################
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (1000000, rlimit[1]))
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #######################################

    args = parse_args(['--dataset', 'DUTS', '-s', 'test', '-t', '40'])
    # args = parse_args(['--dataset', 'MSRA-10k', '-t', '40'])
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    # define a CNN model
    ModelCNN = Solver(None, None, config=args, save_fold=None)
    # transforms对PIL.Image对象进行变换，而Compose将多个transforms操作组合成一个列表，共同对PIL.Image对象进行转换操作.
    transform = transforms.Compose([transforms.Resize((448, 448)),
                                    transforms.ToTensor()])  # ToTensor convert a PIL.Image or Numpy.ndarray to tensor.
    untransform = transforms.Compose([transforms.ToPILImage()])  # ToPILImage convert a tensor to PIL Image object.
    train_data = MyDataset(txt_path='./EORSSD_test.txt', transform=transform)  # 创建MyDataset实例，数据和标签转换成tensor,
    # train_data = MyDataset(txt_path='./TXT/MSRA-10k.txt', transform=transform)  # 创建MyDataset实例，数据和标签转换成tensor,
    # 因为DataLoader要求待处理的数据和标签都必须是tensor.
    trainloader = DataLoader(train_data, batch_size=1)  # Combines a train_data and a sampler,
    # and provides an iterable over the given dataset; DataLoader的参数drop_last设置了关于最后一批样本的处理方法.
    # loop take one batch samples in iterators and generate the superpixels and graph data for the dataset.
    # superpixel_Dataset = []  # store the superpixels of all images in the dataset.
    graph_Dataset = []  # store the graph data for all images in the dataset.
    # 修改多线程的tensor方式为file_system（默认方式为file_descriptor，受限于open files数量）
    # torch.multiprocessing.set_sharing_strategy('file_system')
    for batch_indx, data in enumerate(trainloader, 0):
        if batch_indx < 15:
            continue

        print('batch_index:', batch_indx)

        # get the inputs; data is a list of [inputs, labels]; tensor type.
        images, labels, img_size = data  # torch tensor: B*C*H*W, dtype, floate32, range:0~1; B*2;
        # print(images.shape)
        # print(labels.shape)
        # print('batch_indx is :', batch_indx)
        # plt.imshow(untransform(labels[0]))
        # plt.show()
        # plt.pause(1)  # pause 1 second, update the plot.
        # print('Image type:', images.dtype)

        # convert tensor to numpy ndarray.
        # tensor和numpy的区别在于：相对应的对tensor的操作函数要求输入的维度是B*C*H*W;
        # 而相对应的对numpy.ndarray的操作函数要求输入的维度是B*H*W*C.
        if not isinstance(images, np.ndarray):  # convert tensor to numpy ndarray
            images = images.permute(0, 2, 3, 1).numpy()  # tensor(B, C, H, W)→tensor(B, H, W, C)→numpy(B, H, W, C)
        if not isinstance(labels, np.ndarray):  # convert tensor to numpy ndarray
            labels = labels.permute(0, 2, 3, 1).numpy()  # tensor(B, C, H, W)→tensor(B, H, W, C)→numpy(B, H, W, C)

        n_images = images.shape[0]  # get B, equal to the num of sample in the current batch.
        # multi threads processing

        graph_sp_datalist = []
        for i in range(n_images):
            graph_sp_datalist.append(
                process_image((images[i], i, n_images, args, True, labels[i], img_size[i], ModelCNN)))

        graph_Dataset.extend(graph_sp_datalist)
        # labels_Dataset.append(labels)
        print('batch %d end!' % batch_indx)
        # Save the graph data and superpixels for the whole data in the dataset. Note: pkl是python的一种存储文件，需要安装python打开.
        with open('%s/%s_%s_%dsp_PyGdata_SNIC100_300_500_knn32_batch%d_RGBCNN_forSpatialPretrain.pkl' % (
                args.out_dir, args.dataset, args.split, args.n_sp_query, batch_indx), 'wb') as f:
            # with open('%s/%s_%dsp_PyGdata_SNIC500_knn32_batch1_forSpatialPretrain.pkl' % (args.out_dir, args.dataset, args.n_sp_query), 'wb') as f:
            # pickle.dump(obj, file, [,protocol])
            pickle.dump(graph_sp_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('%s/%s_%dsp_%s_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp_query, args.split), 'wb') as f:
    #     pickle.dump(superpixel_Dataset, f, protocol=2)

    print('done in {}'.format(datetime.datetime.now() - dt))
