import contextlib
import copy
import os

import numpy as np

from clientmanager.BaseClientManager import BaseClientManager
from core.Runtime import Mode, ModeFactory
from utils.GlobalVarGetter import GlobalVarGetter

START = 1
JOIN = 2


class LargeScale(Mode):
    client_dict = {}
    flag = 0
    c_num = 0
    true_num = 0
    total_num = 0
    true_client_list = []

    def __init__(self, client, true_num, real_mode='thread', exchange_logic=None, shared_values=None):
        super().__init__(client)
        if exchange_logic is None:
            exchange_logic = {}
        if shared_values is None:
            shared_values = {}
        LargeScale.c_num += 1
        LargeScale.true_num = true_num
        LargeScale.total_num = GlobalVarGetter.get()["config"]["global"]["client_num"] if LargeScale.total_num == 0 else LargeScale.total_num

        LargeScale.client_dict[client.client_id] = client
        if LargeScale.c_num == LargeScale.total_num:
            id_list = np.array_split(list(range(LargeScale.total_num)), true_num)
            if isinstance(real_mode, dict):
                real_params = real_mode['params'] if 'params' in real_mode else {}
                real_mode = real_mode['path']
            else:
                real_params = None
            multi_gpu = GlobalVarGetter.get()["config"]["global"]["multi_gpu"] if "multi_gpu" in GlobalVarGetter.get()["config"]["global"] else False
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                dev_list = BaseClientManager.get_client_dev_list(true_num, multi_gpu)
                for i in range(true_num):
                    LargeScale.true_client_list.append(
                        ModeFactory.create_mode_instance(
                            ClientWrapper([LargeScale.client_dict[c_id] for c_id in id_list[i]], dev_list[i], exchange_logic, shared_values), real_mode, real_params
                        )
                    )

    def run(self):
        pass

    def start(self):
        if not LargeScale.flag & START:
            LargeScale.flag = LargeScale.flag | START
            for c in LargeScale.true_client_list:
                c.start()

    def join(self):
        if not LargeScale.flag & JOIN:
            LargeScale.flag = LargeScale.flag | JOIN
            for c in LargeScale.true_client_list:
                c.join()


class ClientWrapper:
    def __init__(self, client_list, dev, exchange_logic, shared_key):
        self.model = None
        self.exchange_logic = exchange_logic
        self.dev = dev
        self.id_list = [c.client_id for c in client_list]
        self.client_dict = {c.client_id: c for c in client_list}
        self.untraining_params = {}
        self.shared_key = shared_key
        self.selected_event_dict = {c.client_id: c.event for c in client_list}
        self.stop_event_dict = {c.client_id: c.stop_event for c in client_list}
        self.saved_values = {i: {} for i in self.id_list}
        self.shared_values = {}

    def run(self):
        self.share_memory()
        self._run()

    def share_memory(self):
        for i, client_ins in enumerate(self.client_dict.values()):
            client_ins.dev = self.dev
            if i == 0:
                client_ins.init_client()
                self.model = client_ins.model
                self.training_params = client_ins.training_params
                GlobalVarGetter.get()['share_model'] = self.model
                for k in self.shared_key:
                    self.shared_values[k] = getattr(client_ins, k)
                eval(self.exchange_logic['init']) if 'init' in self.exchange_logic else None
            else:
                def create_model():
                    client_ins.model = GlobalVarGetter.get()['share_model']
                    client_ins.training_params = self.training_params
                client_ins.create_model = create_model
                client_ins.init_client()
                for k in self.shared_key:
                    setattr(client_ins, k, self.shared_values[k])

    def _run(self):
        while len(self.id_list):
            for c_id, e, se in zip(self.id_list, self.selected_event_dict.values(), self.stop_event_dict.values()):
                if e.is_set():
                    e.clear()
                    if c_id in self.untraining_params:
                        state_dict = self.client_dict[c_id].model.state_dict()
                        for k in self.untraining_params[c_id]:
                            state_dict[k] = copy.deepcopy(self.untraining_params[c_id][k])
                        self.client_dict[c_id].model.load_state_dict(state_dict)
                    eval(self.exchange_logic['before']) if 'before' in self.exchange_logic else None

                    # stop a client which it must be selected
                    if se.is_set():
                        self.client_dict[c_id].finish_client()
                        se.clear()
                        self.stop_event_dict.pop(c_id)
                        self.id_list.remove(c_id)
                    else:
                        self.client_dict[c_id].local_run()
                        if c_id not in self.untraining_params:
                            self.untraining_params[c_id] = {}
                        state_dict = self.client_dict[c_id].model.state_dict()
                        for k in state_dict:
                            if not self.training_params[k]:
                                self.untraining_params[c_id][k] = copy.deepcopy(state_dict[k])
                        eval(self.exchange_logic['after']) if 'after' in self.exchange_logic else None
