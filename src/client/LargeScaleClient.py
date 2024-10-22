import contextlib
import copy
import os
import sys
from time import sleep

import numpy as np

from client.NormalClient import NormalClient
from clientmanager.BaseClientManager import BaseClientManager
from core.Runtime import Mode, running_mode_for_client
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


def split_dict(data, id_list):
    n = len(data)
    k = len(id_list)
    if k <= 0 or k > n:
        return []
    result = []
    for i, idlst in enumerate(id_list):
        result.append({})
        for id in idlst:
            result[i][id] = data[id]
    return result


class LargeScaleClient(Mode):
    config = None
    dev_dict = {}
    stop_event_dict = {}
    selected_event_dict = {}
    delay_dict = {}
    index_list_dict = {}
    client_list = []
    cid_true_map = {}
    true_cid_map = None
    flag = 0
    c_num = 0

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        LargeScaleClient.c_num += 1
        if LargeScaleClient.true_cid_map is None:
            LargeScaleClient.config = config
            LargeScaleClient.true_num = config["client_num"] if "client_num" in config else 1
            LargeScaleClient.true_cid_map = [[] for _ in range(LargeScaleClient.true_num)]
            multi_gpu = GlobalVarGetter.get()["config"]["global"]["multi_gpu"]
            LargeScaleClient.total_num = GlobalVarGetter.get()["config"]["global"]["client_num"]
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                LargeScaleClient.dev_list = BaseClientManager.get_client_dev_list(LargeScaleClient.true_num, multi_gpu)
        LargeScaleClient.selected_event_dict[c_id] = selected_event
        LargeScaleClient.stop_event_dict[c_id] = stop_event
        LargeScaleClient.index_list_dict[c_id] = index_list
        LargeScaleClient.delay_dict[c_id] = delay
        if LargeScaleClient.c_num == LargeScaleClient.total_num:
            true_num = LargeScaleClient.true_num
            # transform dict to each true client
            id_list = np.array_split(list(range(LargeScaleClient.total_num)), true_num)
            stop_event_list = split_dict(LargeScaleClient.stop_event_dict, id_list)
            selected_event_list = split_dict(LargeScaleClient.selected_event_dict, id_list)
            delay_list = split_dict(LargeScaleClient.delay_dict, id_list)
            whole_index_list = split_dict(LargeScaleClient.index_list_dict, id_list)
            dev_list = LargeScaleClient.dev_list
            for i in range(true_num):
                LargeScaleClient.client_list.append(_TrueLargeScaleClient(id_list[i], stop_event_list[i], selected_event_list[i], delay_list[i], whole_index_list[i], config, dev_list[i]))

    def start(self):
        if not LargeScaleClient.flag & 1:
            LargeScaleClient.flag = LargeScaleClient.flag | 1
            for c in LargeScaleClient.client_list:
                c.start()

    def join(self):
        if not LargeScaleClient.flag & 2:
            LargeScaleClient.flag = LargeScaleClient.flag | 2
            for c in LargeScaleClient.client_list:
                c.join()

    def run(self):
        pass


class _TrueLargeScaleClient(running_mode_for_client()):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__()
        self.training_params = None
        self.model = None
        self.whole_index_list = index_list
        self.delay_list = delay
        self.selected_event_list = selected_event
        self.stop_event_list = stop_event
        self.id_list = c_id
        self.untraining_params = {}
        self.shared_values = {}
        self.exchange_logic = None
        self.saved_values = {i: {} for i in self.id_list}
        self.inner_client_dict = self.create_inner_client_dict(config, dev)

    def create_inner_client_dict(self, config, dev):
        inner_client = ModuleFindTool.find_class_by_path(config['inner_client']['path'])
        inner_client_dict = {}
        self.exchange_logic = config['exchange_logic'] if 'exchange_logic' in config else {}
        for i, c_id in enumerate(self.id_list):
            client_ins = inner_client(c_id, self.stop_event_list[c_id], self.selected_event_list[c_id], self.delay_list[c_id], self.whole_index_list[c_id], config, dev)
            client_ins.init_client()
            if i == 0:
                self.model = client_ins.model
                self.training_params = client_ins.training_params
                if 'shared_values' in config:
                    for k in config['shared_values']:
                        self.shared_values[k] = getattr(client_ins, k)
                eval(self.exchange_logic['init']) if 'init' in self.exchange_logic else None
            else:
                del client_ins.model
                client_ins.model = self.model
                client_ins.create_optimizer()
                if 'shared_values' in config:
                    for k in config['shared_values']:
                        setattr(client_ins, k, self.shared_values[k])
            inner_client_dict[c_id] = client_ins
        return inner_client_dict

    def run(self):
        while len(self.id_list):
            for c_id, e, se in zip(self.id_list, self.selected_event_list.values(), self.stop_event_list.values()):
                if e.is_set():
                    e.clear()
                    if c_id in self.untraining_params:
                        state_dict = self.inner_client_dict[c_id].model.state_dict()
                        for k in self.untraining_params[c_id]:
                            state_dict[k] = copy.deepcopy(self.untraining_params[c_id][k])
                        self.inner_client_dict[c_id].model.load_state_dict(state_dict)
                    eval(self.exchange_logic['before']) if 'before' in self.exchange_logic else None

                    # stop a client which it must be selected
                    if se.is_set():
                        self.inner_client_dict[c_id].finish_client()
                        se.clear()
                        self.stop_event_list.pop(c_id)
                        self.id_list.remove(c_id)
                    else:
                        self.inner_client_dict[c_id].local_run()
                        if c_id not in self.untraining_params:
                            self.untraining_params[c_id] = {}
                        state_dict = self.inner_client_dict[c_id].model.state_dict()
                        for k in state_dict:
                            if not self.training_params[k]:
                                self.untraining_params[c_id][k] = copy.deepcopy(state_dict[k])
                        eval(self.exchange_logic['after']) if 'after' in self.exchange_logic else None
