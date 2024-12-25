from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ServerHandler import ContentDispatcher
from scheduler.SyncScheduler import SyncScheduler


class SemiAsyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        self.group_manager = self.global_var['group_manager']
        self.group_num = self.group_manager.update()

    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(GroupUpdater())
         .set_next(GroupClientSelector())
         .set_next(ContentDispatcher())
         .set_next(GroupUpdateWaiter()))


class GroupUpdater(Handler):
    def _handle(self, request):
        return request


class GroupClientSelector(Handler):
    def __init__(self):
        super().__init__()
        self.handler = InnerGroupClientSelector()
        self.first_run = False

    def _handle(self, request):
        scheduler = request.get('scheduler')
        group_manager = scheduler.group_manager
        total_selected_clients = []
        if self.first_run is False:
            for i in range(group_manager.get_group_num()):
                for j in group_manager.get_group_list()[i]:
                    scheduler.download_item(j, "group_id", i)
                client_list = group_manager.get_group_list()[i]
                selected_clients = self.handler.handle(
                    {'client_list': client_list, 'group_id': i, 'scheduler': scheduler})
                group_manager.group_client_num_list[i] = len(selected_clients)
                group_manager.network_list[i] = scheduler.server_weights
                total_selected_clients.extend(selected_clients)
            self.first_run = True
        else:
            group_id = scheduler.group_ready_num
            client_list = group_manager.get_group_list()[group_id]
            selected_clients = self.handler.handle(
                {'group_id': group_id, 'client_list': client_list, 'scheduler': scheduler})
            group_manager.group_client_num_list[group_id] = len(selected_clients)
            total_selected_clients.extend(selected_clients)
        request['selected_clients'] = total_selected_clients
        return request


class InnerGroupClientSelector(Handler):
    def _handle(self, request):
        client_list = request.get('client_list')
        group_id = request.get('group_id')
        scheduler = request.get('scheduler')
        training_status = scheduler.message_queue.get_training_status()
        client_list = [client_id for client_id in client_list if
                       client_id not in training_status or not training_status[client_id]]
        selected_clients = scheduler.schedule_caller.schedule(client_list)
        print(f'group {group_id} selected_clients: {selected_clients}')
        return selected_clients


class GroupUpdateWaiter(Handler):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        scheduler.queue_manager.receive(scheduler.group_manager.group_client_num_list)
        scheduler.group_ready_num = scheduler.queue_manager.group_ready_num
        print(f'group_ready_num: {scheduler.group_ready_num}', scheduler.group_manager.group_client_num_list)
        return request
