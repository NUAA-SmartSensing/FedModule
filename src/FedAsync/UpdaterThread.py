import collections
import threading
import time
import copy
from torch.utils.data import DataLoader

import torch.utils.data


class UpdaterThread(threading.Thread):
    def __init__(self, queue, server_thread_lock, t, c_t, g_n, async_client_manager, s_e, test_data):
        threading.Thread.__init__(self)
        self.queue = queue
        self.server_thread_lock = server_thread_lock
        self.T = t
        self.current_time = c_t
        self.server_network = g_n
        self.async_client_manager = async_client_manager
        self.stop_event = s_e
        self.test_data = test_data

        self.check_in_thread_lock = self.async_client_manager.get_check_in_thread_lock()

        self.event = threading.Event()
        self.event.clear()

        self.sum_delay = 0

        self.accuracy_list = []
        self.loss_list = []

    def run(self):
        for epoch in range(self.T):
            # self.check_in_thread_lock.acquire()
            while True:
                self.check_in_thread_lock.acquire()
                c_r = 0
                # 接收一个client发回的模型参数和时间戳
                if not self.queue.empty():
                    (c_id, client_weights, time_stamp) = self.queue.get()
                    self.sum_delay += (self.current_time.get_time() - time_stamp)
                    print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                          self.current_time.get_time(), "| queue size = ", self.queue.qsize())
                    self.event.set()
                else:
                    (c_id, client_weights, time_stamp) = (0, [], 0)

                if self.event.is_set():
                    # 使用接收的client发回的模型参数和时间戳对全局模型进行更新
                    self.server_thread_lock.acquire()
                    self.update_server_weights(client_weights, time_stamp)
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
        # print("----------------------------------------------------------------------------stop")
        self.async_client_manager.stop_all_clients()
        # print("----------------------------------------------------------------------------stopped")
        # print("Thread count =", threading.activeCount())
        # print(*threading.enumerate(), sep="\n")

    def update_server_weights(self, client_weights: collections.OrderedDict, time_stamp):
        # if s_type == "Constant":
        #     s = 1
        # elif s_type == "Polynomial":
        #     s = float(1 / ((self.current_time.get_time() - time_stamp + 1) ** a))
        # elif s_type == "Hinge":
        b = 60
        a = 0.1
        alpha = 0.1
        r = 1
        if (self.current_time.get_time() - time_stamp) <= b:
            s = 1
        else:
            s = float(1 / ((a * (self.current_time.get_time() - time_stamp - b)) + 1))
        # else:
        #     s = 1
        #     print("Error in s-type!!!!")
        # r = 1
        alpha = alpha * s * r
        updated_parameters = {}
        server_weights = copy.deepcopy(self.server_network.state_dict())
        for key, var in client_weights.items():
            updated_parameters[key] = var.clone()
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        for key, var in server_weights.items():
            updated_parameters[key] = (alpha * updated_parameters[key] + (1 - alpha) * server_weights[key])
            # updated_parameters[key] = (updated_parameters[key] + server_weights[key]) / 2
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

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list
