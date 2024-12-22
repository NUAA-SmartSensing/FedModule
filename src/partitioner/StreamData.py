import random
from collections import Counter

from core.MessageQueue import MessageQueueFactory
from partitioner.AbstractPartitioner import AbstractPartitioner
from utils.GlobalVarGetter import GlobalVarGetter
from utils.IID import generate_iid_data, generate_non_iid_data


class _StreamData(AbstractPartitioner):
    def __init__(self, config):
        super().__init__(config)
        self.class_num_per_task = self.config.get('class_num', 2)
        self.task_num = self.config.get('task_num', 0)
        self.random = self.config.get('random', True)

    def generate_data(self, iid_config, labels, client_num, dataset):
        if self.task_num == 0:
            self.task_num = len(set(labels)) // self.class_num_per_task


class ClientTaskPartition(_StreamData):
    r"""
        This class only divides the whole dataset into several independent tasks, not mapping the classes' label.
        The generation process is as follows:
        1. Randomly shuffle the label list if the random is True.
        2. according to the iid_config, generate the index list for each client.
        3. For each client, divide the data into several tasks, and each task contains several classes.
    """

    def __init__(self, config):
        super().__init__(config)
        self.label_list = None
        self.is_sort = self.config.get('sort', True)
        self.duplication = self.config.get('duplication', False)

    def generate_data(self, iid_config, labels, client_num, dataset):
        super().generate_data(iid_config, labels, client_num, dataset)
        index_list = self.generate_data_for_training(labels, client_num)
        return index_list

    def generate_data_for_training(self, labels, client_num):
        class_num_per_task = self.class_num_per_task
        task_num = self.task_num
        classes = set(labels)
        self.label_list = split_list(list(classes), class_num_per_task, task_num, self.is_sort, self.duplication)
        index_list = split_data(self.config['iid'], labels, client_num)
        index_list_for_task = []
        for i in range(client_num):
            index_list_for_task.append([])
            for j in range(task_num):
                res = [index for index in index_list[i] if labels[index] in self.label_list[j]]
                index_list_for_task[i].append(res)
        print_dist(index_list_for_task, labels)
        return index_list_for_task


class ClientTaskPartitionWithMapping(ClientTaskPartition):
    r"""
        The class is used to divide a whole dataset into several independent tasks.
        And each task contains several classes. classes' label are mapped to 0, 1, 2, ...
        The generation process is as follows:
        1. Randomly shuffle the label list if the random is True.
        2. according to the iid_config, generate the index list for each client.
        3.1 For each client, divide the data into several tasks, and each task contains several classes.
        3.2 Mapping the classes' label to 0, 1, 2, ...
    """

    def __init__(self, config):
        super().__init__(config)
        self.label_mapping = None

    def generate_data(self, iid_config, labels, client_num, dataset):
        _StreamData.generate_data(self,iid_config, labels, client_num, dataset)
        index_list, label_mapping = self.generate_data_for_training(labels, client_num)
        mq = MessageQueueFactory.create_message_queue()
        for i in range(client_num):
            mq.put_into_downlink(i, "label_mapping", label_mapping)
        GlobalVarGetter.get()["label_mapping"] = label_mapping
        return index_list

    def generate_data_for_training(self, labels, client_num):
        index_list_for_task = super().generate_data_for_training(labels, client_num)
        self.label_mapping = {}
        for label_list in self.label_list:
            for i, label in enumerate(label_list):
                self.label_mapping[label] = i
        print("label mapping", self.label_mapping)
        return index_list_for_task


class TaskClientPartition(ClientTaskPartition):
    def generate_data_for_training(self, labels, client_num):
        class_num_per_task = self.class_num_per_task
        task_num = self.task_num
        classes = set(labels)
        label_list_for_task = split_list(list(classes), class_num_per_task, task_num, self.is_sort, self.duplication)
        self.label_list = label_list_for_task
        index_label_list_for_task = []
        for label_list in label_list_for_task:
            index_label_list_for_task.append([i for i, label in enumerate(labels) if label in label_list])
        index_lists = [[] for _ in range(client_num)]
        for index_label_list in index_label_list_for_task:
            results = split_data(self.config['iid'], labels[index_label_list], client_num)
            index_list = [[index_label_list[i] for i in result] for result in results]
            for i in range(client_num):
                index_lists[i].append(index_list[i])
        print_dist(index_lists, labels)
        return index_lists


class TaskClientPartitionWithMapping(TaskClientPartition):
    def __init__(self, config):
        super().__init__(config)
        self.label_mapping = None

    def generate_data(self, iid_config, labels, client_num, dataset):
        _StreamData.generate_data(self, iid_config, labels, client_num, dataset)
        index_list, label_mapping = self.generate_data_for_training(labels, client_num)
        mq = MessageQueueFactory.create_message_queue()
        for i in range(client_num):
            mq.put_into_downlink(i, "label_mapping", label_mapping)
        GlobalVarGetter.get()["label_mapping"] = label_mapping
        return index_list

    def generate_data_for_training(self, labels, client_num):
        index_list_for_task = super().generate_data_for_training(labels, client_num)
        self.label_mapping = {}
        for label_list in self.label_list:
            for i, label in enumerate(label_list):
                self.label_mapping[label] = i
        print("label mapping", self.label_mapping)
        return index_list_for_task


def split_list(rsc, sublist_length, sublist_num, is_sort=True, allow_duplicates=False):
    if is_sort:
        rsc = sorted(rsc)
    else:
        random.shuffle(rsc)
    if not allow_duplicates and len(rsc) < sublist_length * sublist_num:
        raise Exception(
            "The length of the resource is less than the length of the sublist. Please open the allow_duplicates switch.")
    result = []
    start = 0
    for i in range(sublist_num):
        end = start + sublist_length
        if allow_duplicates and end > len(rsc):
            end = end - len(rsc)
            sublist = rsc[start:] + rsc[:end]
        else:
            sublist = rsc[start:end]
        result.append(sublist)
        start = end
    return result


def split_data(iid_config, labels, nums):
    if isinstance(iid_config, bool):
        index_list = generate_iid_data(labels, nums)
    else:
        index_list = generate_non_iid_data(iid_config, labels, nums)
    return index_list


def print_dist(index_lists, labels):
    for i, client_index_list in enumerate(index_lists):
        total = sum([len(index_list) for index_list in client_index_list])
        print(f'({i}: {total}, ', end='')
        print('{', end='')
        for task_id, index_list in enumerate(client_index_list):
            print(f"task{task_id}: ", end='')
            counts = Counter(labels[index_list])
            counts = dict(counts)
            for k, v in counts.items():
                print(f" {k}: {v}",end=',')
            print(end=' | ')
        print('}')
