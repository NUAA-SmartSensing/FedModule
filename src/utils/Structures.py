from collections import deque


class BidirectionalMappedQueue:
    def __init__(self):
        self.group_to_data = {}
        self.id_to_group = {}

    def add_data(self, data_id, group_num, data):
        if group_num not in self.group_to_data:
            self.group_to_data[group_num] = deque()

        self.group_to_data[group_num].append((data_id, data))
        self.id_to_group[data_id] = group_num

    def get_group(self, group_num):
        return self.group_to_data[group_num]

    def get_group_for_id(self, data_id):
        return self.id_to_group[data_id]


class BidirectionalMappedList:
    def __init__(self):
        self.group_to_data = {}
        self.id_to_group = {}

    def add_data(self, data_id, group_num, data):
        if group_num not in self.group_to_data:
            self.group_to_data[group_num] = []

        self.group_to_data[group_num].append((data_id, data))
        self.id_to_group[data_id] = group_num

    def get_group(self, group_num):
        return self.group_to_data[group_num]

    def get_group_for_id(self, data_id):
        return self.id_to_group[data_id]
