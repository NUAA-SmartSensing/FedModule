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
        # 获取全局data_proxy
        global_var = GlobalVarGetter.get()
        data_proxy = global_var.get('data_proxy', None)
        for i, c_id in enumerate(id_list):
            # 检查Client类是否支持data_proxy参数
            try:
                client = client_class(c_id, stop_event_list[i], selected_event_list[i], client_staleness_list[i],
                                      index_list[i], client_config, dev[i], data_proxy=data_proxy)
            except TypeError:
                client = client_class(c_id, stop_event_list[i], selected_event_list[i], client_staleness_list[i],
                                      index_list[i], client_config, dev[i])
            client_list.append(ModeFactory.create_mode_instance(client, mode, params))
        return client_list

    @staticmethod
    def create_client(c_id, stop_event, selected_event, client_staleness, index_list, client_config, dev, total_config=None):
        if not total_config:
            total_config = GlobalVarGetter.get()['config']
        mode, params = running_mode(total_config)
        client_class = ModuleFindTool.find_class_by_path(total_config["client"]["path"])
        global_var = GlobalVarGetter.get()
        data_proxy = global_var.get('data_proxy', None)
        try:
            client = client_class(c_id, stop_event, selected_event, client_staleness, index_list, client_config, dev, data_proxy=data_proxy)
        except TypeError:
            client = client_class(c_id, stop_event, selected_event, client_staleness, index_list, client_config, dev)
        return ModeFactory.create_mode_instance(client, mode, params)  # instance
