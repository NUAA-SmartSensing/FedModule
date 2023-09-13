import pandas as pd
import torch

from dataset.BaseDataset import BaseDataset
from utils.DataReader import CustomDataset
from utils.DatasetUtils import *


class PAMAP2(BaseDataset):
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
def preprocess(dataset_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/"), window_size=171, overlap_rate=0.5, split_rate=(8, 2),
               validation_subjects={105}, z_score=True, save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/PAMAP2_Dataset/datasets/")):
    if os.path.exists(save_path):
        xtrain, xtest, ytrain, ytest = np.load(os.path.join(save_path, 'PAMAP2/x_train.npy')), \
                                       np.load(os.path.join(save_path, 'PAMAP2/x_test.npy')), \
                                       np.load(os.path.join(save_path, 'PAMAP2/y_train.npy')), \
                                       np.load(os.path.join(save_path, 'PAMAP2/y_test.npy'))
        return torch.from_numpy(xtrain).float().unsqueeze(1), torch.from_numpy(xtest).float().unsqueeze(1), torch.from_numpy(ytrain).long(), torch.from_numpy(ytest).long()
    '''
        dataset_dir: 源数据目录 : str
        window_size: 滑窗大小 : int
        overlap_rate: 滑窗重叠率 : float in [0，1）
        split_rate: 平均法分割验证集，表示训练集与验证集比例。优先级低于"VALIDATION_SUBJECTS": tuple
        validation_subjects: 留一法分割验证集，表示验证集所选取的Subjects : set
        z_score: 标准化 : bool
        save_path: 预处理后npy数据保存目录 : str
    '''

    print(
        "\n原数据分析：共12个活动，文件包含9个受试者收集的数据，切分数据集思路可以采取留一法，选取n个受试者数据作为验证集。\n")
    print('预处理思路：提取有效列，重置活动label，数据降采样1/3，即100Hz -> 33.3Hz，进行滑窗，缺值填充，标准化等方法\n')

    #  保证验证选取的subjects无误
    if validation_subjects:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (validation_subjects))
        for each in validation_subjects:
            assert each in set([*range(101, 110)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(split_rate)))

    # 下载数据集
    download_dataset(
        dataset_name='PAMAP2',
        file_url='http://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], []  # train-test-data, 用于存放最终数据
    category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]))  # 12分类所对应的实际label，对应readme.pdf

    dir = os.path.join(dataset_dir, "PAMAP2_Dataset/Protocol")
    filelist = os.listdir(dir)
    os.chdir(dir)
    print('Loading subject data')
    for file in filelist:

        subject_id = int(file.split('.')[0][-3:])
        print('     current subject: 【%d】' % (subject_id), end='')
        print('   ----   Validation Data' if subject_id in validation_subjects else '')

        content = pd.read_csv(file, sep=' ', usecols=[1] + [*range(4, 16)] + [*range(21, 33)] + [
            *range(38, 50)])  # 取出有效数据列, 第2列为label，5-16，22-33，39-50都是可使用的传感数据
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()  # 线性插值填充nan

        # 降采样 1/3， 100Hz -> 33.3Hz
        data = content[::3, 1:]  # 数据 （n, 36)
        label = content[::3, 0]  # 标签

        data = data[label != 0]  # 去除0类
        label = label[label != 0]

        for label_id in range(12):
            true_label = category_dict[label_id]
            cur_data = sliding_window(array=data[label == true_label], windowsize=window_size, overlaprate=overlap_rate)

            # 两种分割验证集的方法 [留一法 or 平均法]
            if validation_subjects:  # 留一法
                # 区分训练集 & 验证集
                if subject_id not in validation_subjects:  # 训练集
                    xtrain += cur_data
                    ytrain += [label_id] * len(cur_data)
                else:  # 验证集
                    xtest += cur_data
                    ytest += [label_id] * len(cur_data)
            else:  # 平均法
                trainlen = int(len(cur_data) * split_rate[0] / sum(split_rate))  # 训练集长度
                testlen = len(cur_data) - trainlen  # 验证集长度
                xtrain += cur_data[:trainlen]
                xtest += cur_data[trainlen:]
                ytrain += [label_id] * trainlen
                ytest += [label_id] * testlen

    os.chdir('../')

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)

    if z_score:  # 标准化
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)

    print(
        '\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s' % (
        xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if save_path:  # 数组数据保存目录
        save_npy_data(
            dataset_name='PAMAP2',
            root_dir=save_path,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )

    return xtrain, xtest, ytrain, ytest
