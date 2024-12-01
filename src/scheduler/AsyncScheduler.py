from core.handlers.Handler import Filter
from core.handlers.ServerHandler import ClientSelector
from scheduler.SyncScheduler import SyncScheduler
from utils.GlobalVarGetter import GlobalVarGetter


# this scheduler schedules clients according to the number of aggregations
class AsyncScheduler(SyncScheduler):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ClientSelectorFilter(), ClientSelector)


# this scheduler schedules clients according to the nums of update which clients update
class AsyncSchedulerWithUpdate(SyncScheduler):
    def schedule(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ClientSelectorFilterWithUpdate(), ClientSelector)


class ClientSelectorFilter(Filter):
    def __init__(self):
        super().__init__()
        self.last_s_time = -1
        config = GlobalVarGetter.get()['config']['server']['scheduler']
        self.schedule_interval = config.get('schedule_interval', 1)
        self.schedule_delay = config.get('schedule_delay', 1)

    def _handle(self, request):
        scheduler = request.get('scheduler')
        current_t = scheduler.current_t.get_time()
        if (current_t - 1) % self.schedule_interval == 0 and current_t != self.last_s_time and current_t <= scheduler.T:
            print("| current_epoch", current_t, "| last schedule time =",
                  self.last_s_time)
            # scheduling according to the number of aggregations.
            if scheduler.queue_manager.size() <= self.schedule_delay:
                print("| queue.size |", scheduler.queue_manager.size(), "<=", self.schedule_delay)
                self.last_s_time = current_t
                return True
            else:
                print("| queue.size |", scheduler.queue_manager.size(), ">", self.schedule_delay)
                print("\n-----------------------------------------------------------------No Schedule")
                return False


class ClientSelectorFilterWithUpdate(ClientSelectorFilter):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        current_t = scheduler.current_t.get_time()
        # 每隔一段时间进行一次schedule
        if scheduler.queue_manager.get.count % self.schedule_interval == 0 and current_t != self.last_s_time:
            print("| current_time |", current_t % self.schedule_interval, "= 0", current_t, "!=",
                  self.last_s_time)
            print("| queue.size |", scheduler.queue_manager.size(), "<= ", self.schedule_delay)
            # scheduling according to the number of received updates
            if scheduler.queue_manager.size() <= self.schedule_delay:
                self.last_s_time = current_t
                return True
            else:
                print("\n-----------------------------------------------------------------No Schedule")
                return False
