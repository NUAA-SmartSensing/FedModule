import copy

import torch.cuda

from utils import ModuleFindTool
from utils.DataReader import DataReader
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import EventFactory


class BaseClientManager:
    def __init__(self, stop_event, config):
        self.stop_event = stop_event
        self.config = config
        self.global_var = GlobalVarGetter().get()
        self.client_list = [] # client实例列表
        self.client_id_list = [] # 每个client对应的client id

        self.multi_gpu = self.global_var["global_config"]["multi_gpu"]
        self.clients_num = self.global_var["global_config"]["client_num"]
        self.batch_size = self.global_var["client_config"]["batch_size"]
        self.client_staleness_list = self.global_var["client_config"]["stale_list"]
        self.epoch = self.global_var["client_config"]["epochs"]
        self.client_config = self.global_var["client_config"]
        self.current_time = self.global_var["current_t"]
        self.schedule_t = self.global_var["schedule_t"]
        self.dataset = self.global_var["dataset"].get_train_dataset()
        self.index_list = self.global_var["dataset"].get_index_list() # 每个client下的数据集index
        self.queue_manager = self.global_var["queue_manager"]
        self.print_lock = self.global_var["print_lock"]
        self.init_weights = copy.deepcopy(self.global_var["server_network"].state_dict())

        self.client_class = ModuleFindTool.find_class_by_path(self.global_var["client_config"]["path"])
        self.selected_event_list = [EventFactory.create_Event() for _ in range(self.clients_num)]

    def start_all_clients(self):
        self.init_clients()
        # 启动clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Start clients:")
        for client in self.client_list:
            client.start()

    def stop_all_clients(self):
        # 终止所有client线程
        self.stop_event.set()
        for i in range(self.clients_num):
            self.selected_event_list[i].set()

    def init_clients(self):
        data_reader = DataReader(self.dataset)
        mode, dev_num, dev_total, dev_mem_list = self.get_running_mode()
        # 初始化clients
        mem_total = 0
        ratio_list = []
        res_client = self.clients_num
        dev_idx = dev_mem_list.index(max(dev_mem_list)) # 剩余内存最大的显卡
        dev_str = f'cuda:' + str(dev_idx)
        if mode == 0: # 多gpu下，根据剩余显存分配client到gpu device
            for i in dev_mem_list:
                mem_total += i
            for i in range(len(dev_mem_list)-1):
                c_num = int(dev_mem_list[i] / mem_total * self.clients_num) # 比例乘以总数
                res_client = res_client - c_num
                ratio_list = ratio_list + [f'cuda:{i}' for _ in range(c_num)]
            ratio_list = ratio_list + [f'cuda:{len(dev_mem_list)-1}' for _ in range(res_client)] #剩余塞到最后一个显卡上
        for i in range(self.clients_num):
            if mode == 0:
                dev = ratio_list[i]
            elif mode == 1:
                # 单显卡模式下自动选择适合的显卡
                dev = dev_str
            else:
                dev = 'cpu'
            client_delay = self.client_staleness_list[i]
            self.client_list.append(self.client_class(i, self.stop_event, self.selected_event_list[i], client_delay, data_reader.total_data, self.index_list[i], self.client_config, dev)) # 实例化
            self.client_id_list.append(i)

    def get_running_mode(self):
        dev_num = 0
        dev_total = 0
        dev_list = []
        # 0: 多gpu，1：单gpu，2：cpu
        if torch.cuda.is_available():
            if self.multi_gpu:
                mode = 0
                dev_num = 0
                dev_total = torch.cuda.device_count()
                for i in range(dev_total):
                    device = torch.device(f'cuda:{i}')
                    dev_list.append(torch.cuda.get_device_properties(device).total_memory-torch.cuda.memory_allocated(device)) # 每个显卡的剩余内存
            else:
                mode = 1
        else:
            mode = 2
        return mode, dev_num, dev_total, dev_list

    def get_client_list(self):
        client_list = self.client_list
        return client_list

    def get_client_id_list(self):
        return self.client_id_list
