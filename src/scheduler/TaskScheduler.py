from core.handlers.Handler import Handler
from core.handlers.ServerHandler import ClientSelector
from scheduler.SyncScheduler import SyncScheduler
from utils.GlobalVarGetter import GlobalVarGetter


class GlobalTaskScheduler(SyncScheduler):
    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(TaskHandler(), ClientSelector)


class TaskHandler(Handler):
    def __init__(self):
        super().__init__()
        config = GlobalVarGetter.get()['config']['server']['scheduler']
        self.task_num = config.get("task_num", 1)
        self.task_interval = config.get("task_interval", 1)
        self.task_id = 0
        self.total_epoch = 0

    def _handle(self, request):
        scheduler = request.get('scheduler')
        if self.total_epoch % self.task_interval == 0:
            print(f"Global changes task to {self.task_id}")
            scheduler.message_queue.put_into_downlink("all", "task_id", self.task_id)
            self.task_id = (self.task_id + 1) % self.task_num
        self.total_epoch += 1
        return request
