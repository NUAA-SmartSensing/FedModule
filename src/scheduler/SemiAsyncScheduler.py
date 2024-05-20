from scheduler.SyncScheduler import SyncScheduler


class SemiAsyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        self.group_manager = self.global_var['group_manager']
        self.group_num = self.group_manager.update()

    def schedule(self):
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
        if self.check_group_update():
            self.group_num = self.group_manager.update()
        print("| current_epoch |", current_time)
        # the first epoch is to start all groups
        if current_time == 1:
            print("starting all groups")
            for i in range(self.group_manager.get_group_num()):
                for j in self.group_manager.get_group_list()[i]:
                    self.message_queue.put_into_downlink(j, "group_id", i)
                print(f"\nbegin select group {i}")
                selected_clients = self.client_select(i)
                # Store the number of clients scheduled.
                self.group_manager.group_client_num_list.append(len(selected_clients))
                # Global storage of model lists for each group.
                self.group_manager.network_list.append(self.server_weights)
                self.notify_client(selected_clients, current_time, schedule_time)
        else:
            print(f"\nbegin select group {self.group_ready_num}")
            selected_clients = self.client_select(self.group_ready_num)
            self.group_manager.group_client_num_list[self.group_ready_num] = len(selected_clients)
            self.notify_client(selected_clients, current_time, schedule_time)
        # wait for all update from clients of the same group
        self.queue_manager.receive(self.group_manager.group_client_num_list)
        self.group_ready_num = self.queue_manager.group_ready_num

    def client_select(self, group_num, *args, **kwargs):
        client_list = self.group_manager.get_group_list()[group_num]
        selected_clients = self.schedule_caller.schedule(client_list)
        return selected_clients

    def check_group_update(self, *args, **kwargs):
        return False
