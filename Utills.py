import torch
import numpy as np
from torchvision import transforms



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
