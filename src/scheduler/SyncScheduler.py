import time

from scheduler.BaseScheduler import BaseScheduler


class SyncScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

    def run(self):
        while self.current_t.get_time() <= self.T:
            # Scheduling is performed periodically.
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            self.schedule()
            # Notifying the updater to aggregate weights.
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)

    def schedule(self):
        r"""
            schedule the clients
        """
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
        selected_client = self.client_select()
        self.notify_client(selected_client, current_time, schedule_time)
        # Waiting for all clients to upload their updates.
        self.queue_manager.receive(len(selected_client))

    def notify_client(self, selected_client, current_time, schedule_time):
        print(f"| current_epoch {current_time} |. Begin client select")
        print("\nSchedulerThread select(", len(selected_client), "clients):")
        for client_id in selected_client:
            print(client_id, end=" | ")
            # Sending the server's model parameters and timestamps to the clients
            self.send_weights(client_id, current_time, schedule_time)
            # Starting a client thread
            self.selected_event_list[client_id].set()
        print("\n-----------------------------------------------------------------Schedule complete")
        self.schedule_t.time_add()
