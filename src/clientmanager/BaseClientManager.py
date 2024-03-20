from abc import abstractmethod

import torch.cuda


class BaseClientManager:
    def __init__(self, whole_config):
        self.client_list = []  # client实例列表
        self.client_id_list = []  # 每个client对应的client id
        self.whole_config = whole_config

    @abstractmethod
    def start_all_clients(self):
        pass

    @abstractmethod
    def stop_all_clients(self):
        pass

    @abstractmethod
    def stop_client_by_id(self, client_id):
        pass

    @abstractmethod
    def create_and_start_new_client(self, client_delay, dev='cpu'):
        pass

    @abstractmethod
    def client_join(self):
        pass

    def get_client_list(self):
        return self.client_list

    def get_client_id_list(self):
        return self.client_id_list

    @staticmethod
    def __get_running_mode(multi_gpu):
        dev_num = 0
        dev_total = 0
        dev_list = []
        # 0: 多gpu，1：单gpu，2：cpu
        if torch.cuda.is_available():
            dev_num = 0
            dev_total = torch.cuda.device_count()
            for i in range(dev_total):
                device = torch.device(f'cuda:{i}')
                dev_list.append(torch.cuda.mem_get_info(device)[0])  # 每个显卡的剩余内存
            if multi_gpu:
                mode = 0
            else:
                mode = 1
        else:
            mode = 2

        return mode, dev_num, dev_total, dev_list

    @staticmethod
    def get_client_dev_list(clients_num, multi_gpu):
        mode, dev_num, dev_total, dev_mem_list = BaseClientManager.__get_running_mode(multi_gpu)
        # 初始化clients
        mem_total = 0
        ratio_list = []
        res_client = clients_num
        print("Training Mode: ", end='')
        if mode == 0:
            # 多gpu下，根据剩余显存多少分配client到gpu device
            print("Mlti-GPU-Mode \nGPU devices num:", dev_total)
            for i in dev_mem_list:
                mem_total += i
            for i in range(len(dev_mem_list) - 1):
                c_num = int(dev_mem_list[i] / mem_total * clients_num)  # 比例乘以总数
                res_client = res_client - c_num
                ratio_list = ratio_list + [f'cuda:{i}' for _ in range(c_num)]
            # 剩余未分配的client塞到最后一个显卡上
            ratio_list = ratio_list + [f'cuda:{len(dev_mem_list) - 1}' for _ in range(res_client)]
            return ratio_list
        elif mode == 1:
            # 选择剩余内存最大的显卡
            dev_idx = dev_mem_list.index(max(dev_mem_list))
            dev_str = f'cuda:' + str(dev_idx)
            print("Single-GPU-Mode \nUsing cuda:", dev_idx)
            return [dev_str] * clients_num
        else:
            print("CPU-Only")
            return ['cpu'] * clients_num
