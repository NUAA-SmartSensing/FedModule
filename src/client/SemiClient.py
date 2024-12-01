from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler


class SemiClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0

    def customize_upload(self):
        self.upload_item("group_id", self.group_id)

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(GroupSetter(), UpdateReceiver)


class GroupSetter(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.group_id = client.message_queue.get_from_downlink(client.client_id, "group_id")
        return request
