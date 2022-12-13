from torch.utils.data import Dataset
import os, pickle

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