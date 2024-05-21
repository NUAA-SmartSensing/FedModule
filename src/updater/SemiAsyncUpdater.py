import torch.utils.data

from updater.SyncUpdater import SyncUpdater
from utils import ModuleFindTool


class SemiAsyncUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
        self.group_manager = self.global_var["group_manager"]

        self.accuracy_list = []
        self.loss_list = []
        group_update_class = ModuleFindTool.find_class_by_path(config["group"]["path"])
        self.group_update = group_update_class(self.config["group"]["params"])

    def update_group_weights(self, epoch, update_list):
        model, _ = self.group_update.update_server_weights(epoch, update_list)
        if torch.cuda.is_available():
            for key, var in model.items():
                model[key] = model[key].cuda()
        return model

    def update_server_weights(self, epoch, network_list):
        update_list = []
        for i in range(self.global_var['group_manager'].group_num):
            update_list.append({"weights": network_list[i]})
        super().update_server_weights(epoch+1, update_list)
        print(self.group_manager.epoch_list)

    def get_update_list(self):
        update_list = []
        # receive all updates
        while not self.queue_manager.empty(self.queue_manager.group_ready_num):
            update_list.append(self.queue_manager.get(self.queue_manager.group_ready_num))
        self.group_manager.network_list[self.queue_manager.group_ready_num] = self.update_group_weights(self.current_time.get_time(),
                                                                                                        update_list)
        self.group_manager.epoch_list[self.queue_manager.group_ready_num] += 1
        return self.group_manager.network_list
