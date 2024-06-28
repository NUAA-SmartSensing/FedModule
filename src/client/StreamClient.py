from torch.utils.data import DataLoader

from client.NormalClient import NormalClient
from utils.DatasetUtils import FLDataset


class StreamClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_num = config["task_num"]
        self.task_interval = config["task_interval"] if "task_interval" in config else 1
        self.task_id = 0
        self.total_epoch = 0

    def change_task(self):
        self.fl_train_ds = FLDataset(self.train_ds, list(self.index_list)[self.task_id::self.task_num], self.transform, self.target_transform)
        self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.task_id = (self.task_id + 1) % self.task_num

    def local_task(self):
        if self.total_epoch % self.task_interval == 0:
            print(f"Client {self.client_id} change task to {self.task_id}")
            self.change_task()
        super().local_task()
        self.total_epoch += 1
