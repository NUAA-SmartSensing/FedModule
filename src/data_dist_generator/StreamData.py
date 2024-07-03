import numpy as np

from data_dist_generator.AbstractDataGen import AbstractDataGen
from utils.IID import generate_iid_data, generate_non_iid_data


class StreamData(AbstractDataGen):
    def __init__(self, config):
        super().__init__(config)

    def generate_data(self, iid_config, dataset, client_num, train_dataset):
        class_num_per_task = self.config['class_num'] if 'class_num' in self.config else 2
        random = self.config['random'] if 'random' in self.config else True
        unified = self.config['unified'] if 'unified' in self.config else True
        label_list = list(range(dataset.label_min, dataset.label_max + 1))
        if random:
            np.random.shuffle(label_list)
        iid_config = self.config['iid']
        if isinstance(iid_config, bool):
            index_list = generate_iid_data(dataset, client_num)
        else:
            index_list = generate_non_iid_data(iid_config, dataset, client_num, dataset.label_min, dataset.label_max + 1,
                                                    train_dataset)
        index_list_for_task = []
        class_total = dataset.label_max - dataset.label_min + 1
        list_per_label = [{} for _ in range(client_num)]
        for i in range(client_num):
            list_per_label[i] = {j: [] for j in range(dataset.label_min, dataset.label_max + 1)}
            for index in index_list[i]:
                list_per_label[i][train_dataset.targets[index].item()].append(index)
        for i in range(client_num):
            print(f'({i},', end=' ')
            index_list_for_task.append([])
            if not unified:
                np.random.shuffle(label_list)
            for j in range(class_total // class_num_per_task):
                print(f"task{j}: (", end='')
                index_list_for_task[i].append([])
                for k in range(class_num_per_task):
                    print(f"'{label_list[k+j*class_num_per_task]}:{len(list_per_label[i][label_list[k+j*class_num_per_task]])}'", end='')
                    index_list_for_task[i][j] += list_per_label[i][label_list[k+j*class_num_per_task]]
                print(')', end=' | ')
            print()
        return index_list_for_task
