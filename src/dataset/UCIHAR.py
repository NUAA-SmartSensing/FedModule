import pandas as pd
import torch

from dataset.BaseDataset import BaseDataset
from utils.DatasetUtils import *


class UCIHAR(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        xtrain, xtest, ytrain, ytest = preprocess()
        # 获取数据集
        self.train_dataset = CustomDataset(xtrain, ytrain)
        self.test_dataset = CustomDataset(xtest, ytest)
        self.init(clients, self.train_dataset, self.test_dataset)


# The following utility function is borrowed from
# https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/utils.py
# Thanks to xushige for providing this useful preprocessing technique
def preprocess(dataset_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/"),
               save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/UCI_HAR_Dataset/datasets/")):
    if os.path.exists(save_path):
        xtrain, xtest, ytrain, ytest = np.load(os.path.join(save_path, 'UCI_HAR/x_train.npy')), \
                                       np.load(os.path.join(save_path, 'UCI_HAR/x_test.npy')), \
                                       np.load(os.path.join(save_path, 'UCI_HAR/y_train.npy')), \
                                       np.load(os.path.join(save_path, 'UCI_HAR/y_test.npy'))
        return torch.from_numpy(xtrain).float().unsqueeze(1), torch.from_numpy(xtest).float().unsqueeze(
            1), torch.from_numpy(ytrain).long(), torch.from_numpy(ytest).long()
    '''
        dataset_dir: 源数据目录 : str
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''

    print("\n原数据分析：原数据已经指定比例切分好，窗口大小128，重叠率50%\n")
    print("预处理思路：读取数据，txt转numpy array\n")

    # 下载数据集
    download_dataset(
        dataset_name='UCI_HAR',
        file_url='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip',
        dataset_dir=dataset_dir
    )

    dataset = os.path.join(dataset_dir, "UCI_HAR/")

    signal_class = [
        'body_acc_x_',
        'body_acc_y_',
        'body_acc_z_',
        'body_gyro_x_',
        'body_gyro_x_',
        'body_gyro_x_',
        'total_acc_x_',
        'total_acc_y_',
        'total_acc_z_',
    ]

    def xload(X_path):
        x = []
        for each in X_path:
            with open(each, 'r') as f:
                x.append(np.array([eachline.replace('  ', ' ').strip().split(' ') for eachline in f], dtype=np.float32))
        x = np.transpose(x, (1, 2, 0))
        return x

    def yload(Y_path):
        y = pd.read_csv(Y_path, header=None).to_numpy().reshape(-1)
        return y - 1  # label从0开始

    X_train_path = [dataset + '/train/Inertial Signals/' + signal + 'train.txt' for signal in signal_class]
    X_test_path = [dataset + '/test/Inertial Signals/' + signal + 'test.txt' for signal in signal_class]
    Y_train_path = dataset + '/train/y_train.txt'
    Y_test_path = dataset + '/test/y_test.txt'

    X_train = xload(X_train_path)
    X_test = xload(X_test_path)
    Y_train = yload(Y_train_path)
    Y_test = yload(Y_test_path)

    print(
        '\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s' % (
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    if save_path:  # 数组数据保存目录
        save_npy_data(
            dataset_name='UCI_HAR',
            root_dir=save_path,
            xtrain=X_train,
            xtest=X_test,
            ytrain=Y_train,
            ytest=Y_test
        )

    return X_train, X_test, Y_train, Y_test
