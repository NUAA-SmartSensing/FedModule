# The whole file is borrowed from
# https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/utils.py
# Thanks to xushige for providing this useful preprocessing technique
import glob
import os
import shutil
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class CompositeDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        for i in range(1, len(self.lens)):
            self.lens[i] += self.lens[i - 1]

    def __getitem__(self, index):
        pos = 0
        for pos, data_sum in enumerate(self.lens):
            if index < data_sum:
                break
        return self.datasets[pos][index - self.lens[pos]]

    def __len__(self):
        return self.lens[-1]


class FLDataset(Dataset):
    def __init__(self, dataset, idxs, transform=None, target_transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
        self.target_transform = target_transform
        self.data = self.dataset.data
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def change_idxs(self, idxs):
        self.idxs = idxs


def download_dataset(dataset_name, file_url, dataset_dir):
    '''
        数据集下载
    '''
    # 检查是否存在源数据
    if os.path.exists(os.path.join(dataset_dir, dataset_name)):
        return
    print(
        '\n==================================================【 %s 数据集下载】===================================================\n' % (
            dataset_name))
    print('url: %s\n' % (file_url))

    if dataset_name == 'UniMiB-SHAR' and file_url[-4:] == '.git':  # 由于unimib数据集无法直接访问下载，这里我把unimib数据集上传到github进行访问clone
        if os.path.exists(os.path.join(dataset_dir, dataset_name)):
            shutil.rmtree(os.path.join(dataset_dir, dataset_name))
        os.system('git clone %s %s/%s' % (file_url, dataset_dir, dataset_name))

    else:  # 其他数据集
        # download
        dataset_file_path = os.path.join(dataset_dir, 'dataset.zip')
        os.system(f"wget -O {dataset_file_path} {file_url}")

        # unpack
        print("解压数据集")
        while glob.glob(os.path.join(dataset_dir, '*.zip')):
            for file in glob.glob(os.path.join(dataset_dir, '*.zip')):
                for format in ["zip", "tar", "gztar", "bztar", "xztar"]:
                    try:
                        shutil.unpack_archive(filename=file, extract_dir=os.path.join(dataset_dir, dataset_name),
                                              format=format)
                        break
                    except:
                        continue
                os.remove(file)
        print("解压完成")

    print()

    # 检查数据集是否下载完毕
    if not os.path.exists(dataset_dir):
        quit('数据集下载失败，请检查url与网络后重试')


def sliding_window(array, windowsize, overlaprate):
    '''
    array: 二维数据(n, m)，n为时序长度，m为模态轴数。: list or array。
    windowsize: 窗口尺寸
    overlaprate: 重叠率
    '''
    stride = int(windowsize * (1 - overlaprate))  # 计算stride
    times = (len(array) - windowsize) // stride + 1  # 滑窗次数，同时也是滑窗后数据长度
    res = []
    for i in range(times):
        x = array[i * stride: i * stride + windowsize]
        res.append(x)
    return res


def z_score_standard(xtrain, xtest):
    '''
        xtrain: 滑窗后的3维训练集 [n, window_size, modal_leng]。format：array
        xtest: 滑窗后的3维验证集 [n, window_size, modal_leng]。format：array
    '''
    assert xtrain.shape[1:] == xtest.shape[1:]
    window_size, modal_leng = xtrain.shape[1:]
    xtrain_2d, xtest_2d = xtrain.reshape(-1, modal_leng), xtest.reshape(-1, modal_leng)  # reshape成2维，按模态轴进行标准化
    std = StandardScaler().fit(xtrain_2d)
    xtrain_2d, xtest_2d = std.transform(xtrain_2d), std.transform(xtest_2d)
    xtrain, xtest = xtrain_2d.reshape(xtrain.shape[0], window_size, modal_leng), xtest_2d.reshape(xtest.shape[0],
                                                                                                  window_size,
                                                                                                  modal_leng)
    return xtrain, xtest


def build_npydataset_readme(path):
    '''
        构建数据集readme
    '''
    datasets = sorted(os.listdir(path))
    curdir = os.curdir  # 记录当前地址
    os.chdir(path)  # 进入所有npy数据集根目录
    with open('readme.md', 'w') as w:
        for dataset in datasets:
            if not os.path.isdir(dataset):
                continue
            x_train = np.load('%s/x_train.npy' % (dataset))
            x_test = np.load('%s/x_test.npy' % (dataset))
            y_train = np.load('%s/y_train.npy' % (dataset))
            y_test = np.load('%s/y_test.npy' % (dataset))
            category = len(set(y_test.tolist()))
            d = Counter(y_test)
            new_d = {}  # 顺序字典
            for i in range(category):
                new_d[i] = d[i]
            log = '\n===============================================================\n%s\n   x_train shape: %s\n   x_test shape: %s\n   y_train shape: %s\n   y_test shape: %s\n\n共【%d】个类别\ny_test中每个类别的样本数为 %s\n' % (
                dataset, x_train.shape, x_test.shape, y_train.shape, y_test.shape, category, new_d)
            w.write(log)
    os.chdir(curdir)  # 返回原始地址


def save_npy_data(dataset_name, root_dir, xtrain, xtest, ytrain, ytest):
    '''
        dataset_name: 数据集
        root_dir: 数据集保存根目录
        xtrain: 训练数据 : array  [n1, window_size, modal_leng]
        xtest: 测试数据 : array   [n2, window_size, modal_leng]
        ytrain: 训练标签 : array  [n1,]
        ytest: 测试标签 : array   [n2,]
    '''
    path = os.path.join(root_dir, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + '/x_train.npy', xtrain)
    np.save(path + '/x_test.npy', xtest)
    np.save(path + '/y_train.npy', ytrain)
    np.save(path + '/y_test.npy', ytest)
    print('\n.npy数据【xtrain，xtest，ytrain，ytest】已经保存在【%s】目录下\n' % (root_dir))
    build_npydataset_readme(root_dir)
