import collections
import threading
import time
import copy
from torch.utils.data import DataLoader

import torch.utils.data


class UpdaterThread(threading.Thread):
    def __init__(self, queue, server_thread_lock, t, current_t, server_network,
                 async_client_manager, stop_event, test_data, updater_config):
        threading.Thread.__init__(self)
        self.queue = queue
        self.server_thread_lock = server_thread_lock
        self.T = t
        self.current_time = current_t
        self.server_network = server_network
        self.async_client_manager = async_client_manager
        self.stop_event = stop_event
        self.test_data = test_data

        self.check_in_thread_lock = self.async_client_manager.get_check_in_thread_lock()

        self.event = threading.Event()
        self.event.clear()

        self.sum_delay = 0

        self.accuracy_list = []
        self.config = updater_config
        update_module = __import__("update")
        update_file = getattr(update_module, updater_config["update_file"])
        self.update = getattr(update_file, updater_config["update_name"])()

    def run(self):
        for epoch in range(self.T):
            while True:
                self.check_in_thread_lock.acquire()
                c_r = 0
                # 接收一个client发回的模型参数和时间戳
                if not self.queue.empty():
                    # (c_id, client_weights, data_sum, time_stamp) = self.queue.get()
                    update_dict = self.queue.get()
                    c_id = update_dict["client_id"]
                    time_stamp = update_dict["time_stamp"]
                    self.sum_delay += (self.current_time.get_time() - time_stamp)
                    print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                          self.current_time.get_time(), "| queue size = ", self.queue.qsize())
                    self.event.set()
                else:
                    update_dict = {}

                if self.event.is_set():
                    # 使用接收的client发回的模型参数和时间戳对全局模型进行更新
                    self.server_thread_lock.acquire()
                    self.update_server_weights(epoch, update_dict, self.config["params"])
                    self.run_server_test(epoch)
                    self.server_thread_lock.release()
                    self.event.clear()
                    self.check_in_thread_lock.release()
                    time.sleep(0.01)
                    break
                else:
                    # self.event.wait()  # 等待标志位设定
                    self.check_in_thread_lock.release()
                    time.sleep(0.01)

            # self.check_in_thread_lock.release()
            self.current_time.time_add()
            time.sleep(0.01)

        print("Average delay =", (self.sum_delay / self.T))

        # 终止所有client线程
        self.async_client_manager.stop_all_clients()

    def update_server_weights(self, epoch, update_dict, update_param):
        updated_parameters = self.update.update_server_weights(self, epoch, update_dict, update_param)
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        self.server_network.load_state_dict(updated_parameters)

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=100, shuffle=True)
        test_correct = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = self.server_network(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
        accuracy = test_correct / len(dl)
        self.accuracy_list.append(accuracy)
        print('Epoch(t):', epoch, 'accuracy:', accuracy)
        return accuracy

    def get_accuracy_list(self):
        return self.accuracy_list
