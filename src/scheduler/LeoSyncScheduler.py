import time
from utils import ModuleFindTool
from scheduler.BaseScheduler import BaseScheduler
import pandas as pd

class LeoSyncScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        
        self.group_manager = self.global_var['group_manager']
        if self.config["scheduler_access"].split('.')[-1] == 'csv':
            self.global_var['access_group_info'] = pd.read_csv(self.config["scheduler_access"])
        else:
            self.global_var['access_group_info'] = pd.read_excel(self.config["scheduler_access"])

        #具体的选择算法
        schedule_algo_class = ModuleFindTool.find_class_by_path(config["schedule_algo"]["path"])
        self.schedule_algo = schedule_algo_class(config, self.global_var['access_group_info'],**config["schedule_algo"]["params"])
        # self.schedule_caller = ScheduleCaller(self)

        self.global_var['T'], *a = self.schedule_algo.get_status()
        self.T = self.global_var['T']


    def run(self):
        last_s_time = -1
        while self.current_t.get_time() <= self.T:
            # 每隔一段时间进行一次schedule
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            if last_s_time != current_time:
                if current_time > self.T:
                    break
                print("| current_epoch |", current_time)
                print("Begin client select")
                last_s_time = current_time
                # selected client_id 
                sats,self.global_var['end_idx'] = self.schedule_algo.next(self.current_t.get_time())
                # print("Selected sats: ", group_ids)
                selected_client = self.client_select(sats)
                print("\nSchedulerThread select(", len(selected_client), "clients):")
                for client_id in selected_client:
                    print(client_id, end=" | ")
                    # 将server的模型参数和时间戳发给client
                    self.send_weights(client_id, current_time, schedule_time)
                    # 启动一次client线程
                    self.selected_event_list[client_id].set()
                print("\n-----------------------------------------------------------------Schedule complete")
                # 等待所有客户端上传更新
                self.queue_manager.receive(len(selected_client))
                # 通知updater聚合权重
                self.mutex_sem.release()
                self.full_sem.release()
                time.sleep(0.01)

    def client_select(self, sats):
        # client_list = self.global_var['client_id_list']
        selected_clients = self.schedule_caller.schedule(sats)
        return selected_clients