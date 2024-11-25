from core.Runtime import running_mode, ModeFactory
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class ClientFactory:
    @staticmethod
    def create_clients(id_list, stop_event_list, selected_event_list, client_staleness_list, index_list, client_config,
                       dev, total_config=None):
        if not total_config:
            total_config = GlobalVarGetter.get()['config']
        mode, params = running_mode(total_config)
        client_class = ModuleFindTool.find_class_by_path(total_config["client"]["path"])
        client_list = []
        for i, c_id in enumerate(id_list):
            client_list.append(ModeFactory.create_mode_instance(
                client_class(c_id, stop_event_list[i], selected_event_list[i], client_staleness_list[i],
                             index_list[i], client_config, dev[i]), mode, params))  # instance
        return client_list

    @staticmethod
    def create_client(c_id, stop_event, selected_event, client_staleness, index_list, client_config, dev, total_config=None):
        if not total_config:
            total_config = GlobalVarGetter.get()['config']
        mode, params = running_mode(total_config)
        client_class = ModuleFindTool.find_class_by_path(total_config["client"]["path"])
        return ModeFactory.create_mode_instance(
            client_class(c_id, stop_event, selected_event, client_staleness, index_list, client_config, dev), mode, params)  # instance
