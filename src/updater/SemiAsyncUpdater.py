from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ModelTestHandler import ServerTestHandler, ServerPostTestHandler
from core.handlers.ServerHandler import Aggregation, GlobalModelOptimization
from update.UpdateCaller import UpdateCaller
from updater.SyncUpdater import SyncUpdater
from utils import ModuleFindTool


class SemiAsyncUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_manager = self.global_var["group_manager"]
        group_update_class = ModuleFindTool.find_class_by_path(config["group"]["path"])
        self.group_update = group_update_class(self.config["group"]["params"])
        self.group_update_caller = UpdateCaller(self, self.group_update)

    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(InnerGroupUpdateGetter())
            .set_next(InnerGroupAggregation())
            .set_next(GroupUpdateGetter())
            .set_next(GroupAggregation())
            .set_next(GlobalModelOptimization())
            .set_next(ServerTestHandler())
            .set_next(ServerPostTestHandler()))


class InnerGroupUpdateGetter(Handler):
    def _handle(self, request):
        update_list = []
        updater = request.get('updater')
        queue_manager = updater.queue_manager
        # receive all updates
        while not queue_manager.empty(queue_manager.group_ready_num):
            update_list.append(queue_manager.get(queue_manager.group_ready_num))
        request['update_list'] = update_list
        return request


class InnerGroupAggregation(Aggregation):
    def _handle(self, request):
        request = super()._handle(request)
        updater = request.get('updater')
        group_manager = updater.group_manager
        queue_manager = updater.queue_manager
        group_manager.network_list[queue_manager.group_ready_num] = request.get('weights')
        group_manager.epoch_list[queue_manager.group_ready_num] += 1
        return request


class GroupAggregation(Handler):
    def _handle(self, request):
        update_list = request.get('update_list')
        updater = request.get('updater')
        epoch = request.get('epoch')
        global_model, delivery_weights = updater.group_update_caller.update_server_weights(epoch, update_list)
        request['weights'] = global_model
        request['delivery_weights'] = delivery_weights
        global_var = request.get('global_var')
        global_var['delivery_weights'] = delivery_weights
        return request


class GroupUpdateGetter(Handler):
    def _handle(self, request):
        updater = request.get('updater')
        group_manager = updater.group_manager
        update_list = []
        for i in range(group_manager.group_num):
            update_list.append({"weights": group_manager.network_list[i]})
        request['update_list'] = update_list
        return request
