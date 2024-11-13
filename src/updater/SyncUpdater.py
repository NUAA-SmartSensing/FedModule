import wandb

from core.Handler import HandlerChain, ClientUpdateGetter, Aggregation, GlobalModelOptimization, ModelEvaluator, \
    DispatchContentFilter
from updater.BaseUpdater import BaseUpdater
from updater.mixin.MoreTest import TestEachClass, TestMultiTask


class SyncUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

    def run(self):
        for _ in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()

            update_list = self.get_update_list()
            self.server_thread_lock.acquire()
            epoch = self.current_time.get_time()
            self.server_update(epoch, update_list)
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

    def server_update(self, epoch, update_list):
        self.update_server_weights(epoch, update_list)
        acc, loss = self.run_server_test(epoch)
        if self.config['enabled']:
            wandb.log({'accuracy': acc, 'loss': loss})

    def get_update_list(self):
        update_list = []
        # receive all updates
        while not self.queue_manager.empty():
            update_list.append(self.queue_manager.get())
        return update_list


class SyncUpdaterWithDetailedTest(TestEachClass, SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        TestEachClass.__init__(self)
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)


class SyncUpdaterWithTaskTest(TestMultiTask, SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        TestMultiTask.__init__(self)
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)


class HandlerChainUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.model = self.server_network
        self.handler_chain = self.create_handler_chain()

    def run(self):
        for _ in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()

            self.server_thread_lock.acquire()

            self.execute_chain()
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

    def create_handler_chain(self):
        return HandlerChain(
            ClientUpdateGetter(Aggregation(GlobalModelOptimization(DispatchContentFilter(ModelEvaluator())))))

    def execute_chain(self):
        epoch = self.current_time.get_time()
        request = {"epoch": epoch, "updater": self, "global_var": self.global_var,
                   'scheduler': self.global_var['scheduler']}
        self.handler_chain.handle(request)
