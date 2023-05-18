import threading
import time

import torch.utils.data
import wandb
from torch.utils.data import DataLoader

from utils import ModuleFindTool


class AsyncUpdater(threading.Thread):
    def __init__(self, queue, server_thread_lock, t, current_t, schedule_t, server_network,
                 stop_event, test_data, updater_config, receiver_config, global_var):
        threading.Thread.__init__(self)
        self.queue = queue
        self.server_thread_lock = server_thread_lock
        self.T = t
        self.current_time = current_t
        self.schedule_t = schedule_t
        self.server_network = server_network
        self.stop_event = stop_event
        self.test_data = test_data
        self.global_var = global_var
        self.async_client_manager = self.global_var['client_manager']

        self.event = threading.Event()
        self.event.clear()

        self.sum_delay = 0

        self.accuracy_list = []
        self.loss_list = []
        self.config = updater_config
        update_class = ModuleFindTool.find_class_by_path(updater_config["updater_path"])
        self.update = update_class(self.config["params"], self)
        receiver_class = ModuleFindTool.find_class_by_path(receiver_config["receiver_path"])
        self.receiver = receiver_class(queue, receiver_config)

        # loss函数
        if isinstance(updater_config["loss"], str):
            self.loss_func = ModuleFindTool.find_class_by_path(f'torch.nn.functional.{updater_config["loss"]}')
        else:
            self.loss_func = ModuleFindTool.find_class_by_path(
                f'loss.{updater_config["loss"]["loss_file"]}.{updater_config["loss"]["loss_name"]}')

        if isinstance(updater_config["nums"], int):
            self.update_mode = "static"
            self.nums = updater_config["nums"]
        else:
            self.update_mode = "dynamic"
            determineClientUpdateNumClass = ModuleFindTool.find_class_by_path(updater_config["nums"]["nums_path"])
            self.determineClientUpdateNum = determineClientUpdateNumClass(self, updater_config["nums"]["params"])
            self.nums = self.determineClientUpdateNum.init()

    def run(self):
        for epoch in range(self.T):
            while True:
                # 接收一个client发回的模型参数和时间戳
                if not self.queue.empty():
                    # 等待上传
                    self.receiver.receive(self.nums)
                    update_list = []
                    for i in range(self.nums):
                        update_list.append(self.queue.get())
                        c_id = update_list[i]["client_id"]
                        time_stamp = update_list[i]["time_stamp"]
                        self.sum_delay += (self.current_time.get_time() - time_stamp)
                        print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                              self.current_time.get_time(), "| queue size = ", self.queue.qsize())
                    self.event.set()
                else:
                    update_list = []

                if self.event.is_set():
                    # 使用接收的client发回的模型参数和时间戳对全局模型进行更新
                    self.server_thread_lock.acquire()
                    self.update_server_weights(epoch, update_list)
                    self.run_server_test(epoch)
                    self.server_thread_lock.release()
                    self.event.clear()
                    time.sleep(0.01)
                    break
                else:
                    time.sleep(0.01)

            self.current_time.time_add()
            time.sleep(0.01)

        print("Average delay =", (self.sum_delay / self.T))

        # 终止所有client线程
        self.async_client_manager.stop_all_clients()

    def update_server_weights(self, epoch, update_list):
        updated_parameters = self.update.update_server_weights(epoch, update_list)
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        self.server_network.load_state_dict(updated_parameters)

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = self.server_network(inputs)
            _, id = torch.max(outputs.data, 1)
            test_loss += self.loss_func(outputs, labels).item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
        accuracy = test_correct / len(dl)
        loss = test_loss / len(dl)
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
        print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)
        if self.config['enabled']:
            wandb.log({'accuracy': accuracy, 'loss': loss})
        return accuracy, loss

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list
