import numpy as np

from core.MessageQueue import MessageQueueFactory
from data_dist_generator.AbstractDataGen import AbstractDataGen
from utils.GlobalVarGetter import GlobalVarGetter
from utils.IID import generate_iid_data, generate_non_iid_data


class StreamDataByClass(AbstractDataGen):
    r"""
        The class is used to divide a whole class into several tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.label_list = None
        self.class_num_per_task = self.config['class_num'] if 'class_num' in self.config else 2
        self.random = self.config['random'] if 'random' in self.config else True

    def generate_data(self, iid_config, labels, client_num, dataset, train=True):
        if train:
            index_list, label_mapping = self.generate_data_for_training(labels, client_num)
            mq = MessageQueueFactory.create_message_queue()
            for i in range(client_num):
                mq.put_into_downlink(i, "label_mapping", label_mapping)
            GlobalVarGetter.get()["label_mapping"] = label_mapping
        else:
            index_list = self.generate_data_for_test(labels)
        return index_list

    def generate_data_for_training(self, labels, client_num):
        class_num_per_task = self.class_num_per_task
        label_mapping = {}
        label_min = min(labels)
        label_max = max(labels)
        if self.label_list is None:
            self.label_list = list(range(label_min, label_max + 1))
            if self.random:
                np.random.shuffle(self.label_list)
        iid_config = self.config['iid']
        if isinstance(iid_config, bool):
            index_list = generate_iid_data(labels, client_num)
        else:
            index_list = generate_non_iid_data(iid_config, labels, client_num)
        index_list_for_task = []
        class_total = label_max - label_min + 1
        list_per_label = [{} for _ in range(client_num)]
        for i in range(client_num):
            list_per_label[i] = {j: [] for j in range(label_min, label_max + 1)}
            for index in index_list[i]:
                list_per_label[i][labels[index]].append(index)
        for i in range(client_num):
            print(f'({i},', end=' ')
            index_list_for_task.append([])
            for j in range(class_total // class_num_per_task):
                print(f"task{j}: (", end='')
                index_list_for_task[i].append([])
                for k in range(class_num_per_task):
                    label_mapping[self.label_list[k + j * class_num_per_task]] = k
                    print(
                        f"'{self.label_list[k + j * class_num_per_task]}:{len(list_per_label[i][self.label_list[k + j * class_num_per_task]])}'",
                        end='')
                    index_list_for_task[i][j] += list_per_label[i][self.label_list[k + j * class_num_per_task]]
                print(')', end=' | ')
            print()
        return index_list_for_task, label_mapping

    def generate_data_for_test(self, labels):
        if self.label_list is None:
            raise Exception("The label list is None, please generate the training data first.")
        label_min = min(labels)
        label_max = max(labels)
        iid_config = self.config['iid']
        if isinstance(iid_config, bool):
            index_list = generate_iid_data(labels, 1)[0]
        else:
            index_list = generate_non_iid_data(iid_config, labels, 1)[0]
        index_list_for_task = []
        class_total = label_max - label_min + 1
        list_per_label = {j: [] for j in range(label_min, label_max + 1)}
        for index in index_list:
            list_per_label[labels[index]].append(index)
        for i in range(class_total // self.class_num_per_task):
            index_list_for_task.append([])
            for j in range(self.class_num_per_task):
                index_list_for_task[i] += list_per_label[self.label_list[j + i * self.class_num_per_task]]
        return [index_list_for_task]
