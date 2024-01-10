import torch.cuda
import torch.multiprocessing as mp

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import EventFactory


class BaseClientManager:
    def __init__(self, all_config):
        self.all_config = all_config
        self.global_var = GlobalVarGetter.get()
        self.client_list = [] # client实例列表
        self.client_id_list = [] # 每个client对应的client id

        self.multi_gpu = all_config["global"]["multi_gpu"]
        self.clients_num = all_config["global"]["client_num"]
        self.client_staleness_list = all_config["client"]["stale_list"]
        self.index_list = all_config["client"]["index_list"] # 每个client下的数据集index
        self.epoch = all_config["client"]["epochs"]
        self.client_config = all_config["client"]

        self.client_class = ModuleFindTool.find_class_by_path(all_config["client"]["path"])
        self.stop_event_list = [EventFactory.create_Event() for _ in range(self.clients_num)]
        self.selected_event_list = [EventFactory.create_Event() for _ in range(self.clients_num)]
        self.global_var['selected_event_list'] = self.selected_event_list
        self.init_lock = mp.Lock()

    def start_all_clients(self):
        self.__init_clients()
        # 启动clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Start clients")
        for client in self.client_list:
            client.start()

    def __init_clients(self):
        mode, dev_num, dev_total, dev_mem_list = self.get_running_mode()
        # print("available GPU MEMs:",dev_mem_list)
        # 初始化clients
        mem_total = 0
        ratio_list = []
        res_client = self.clients_num
        print("Training Mode: ",end='')
        if mode == 0: 
            # 多gpu下，根据剩余显存多少分配client到gpu device
            print("Mlti-GPU-Mode \nGPU devices num:", dev_total)
            for i in dev_mem_list:
                mem_total += i
            for i in range(len(dev_mem_list)-1):
                c_num = int(dev_mem_list[i] / mem_total * self.clients_num) # 比例乘以总数
                res_client = res_client - c_num
                ratio_list = ratio_list + [f'cuda:{i}' for _ in range(c_num)]
            #剩余未分配的client塞到最后一个显卡上
            ratio_list = ratio_list + [f'cuda:{len(dev_mem_list)-1}' for _ in range(res_client)] 
        elif mode == 1:
            # 选择剩余内存最大的显卡
            dev_idx = dev_mem_list.index(max(dev_mem_list)) 
            dev_str = f'cuda:' + str(dev_idx)
            print("Single-GPU-Mode \nUsing cuda:", dev_idx)
        else:
            print("CPU-Only")
        for i in range(self.clients_num):
            if mode == 0:
                dev = ratio_list[i]
            elif mode == 1:
                # 单显卡模式下自动选择适合的显卡
                dev = dev_str
                # 也可手动选择
                # dev = 'cuda'
            else:
                dev = 'cpu'
            client_delay = self.client_staleness_list[i]
            self.client_list.append(self.client_class(i, self.init_lock, self.stop_event_list[i], self.selected_event_list[i], client_delay, self.index_list[i], self.client_config, dev)) # 实例化
            self.client_id_list.append(i)

    def get_running_mode(self):
        dev_num = 0
        dev_total = 0
        dev_list = []
        # 0: 多gpu，1：单gpu，2：cpu
        if torch.cuda.is_available():
            dev_num = 0
            dev_total = torch.cuda.device_count()
            for i in range(dev_total):
                device = torch.device(f'cuda:{i}')
                dev_list.append(torch.cuda.mem_get_info(device)[0]) # 每个显卡的剩余内存
            if self.multi_gpu:
                mode = 0
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

    def stop_all_clients(self):
        # 终止所有client线程
        for i in range(self.clients_num):
            self.stop_client_by_id(i)

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()

    def client_join(self):
        for i in self.client_list:
            i.join()
