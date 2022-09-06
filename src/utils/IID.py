import utils
from utils.JsonTool import dict_to_list, list_to_dict
from utils.Tools import generate_label_lists_by_step, generate_label_lists, generate_data_lists


def generate_non_iid_data(label_config, data_config, dataset, clients, left, right):
    # 生成label lists
    # 洗牌算法
    shuffle = False
    if "shuffle" in label_config.keys() and not label_config["shuffle"]:
        shuffle = True
    if isinstance(label_config, dict):
        # step
        if "step" in label_config.keys():
            label_lists = generate_label_lists_by_step(label_config["step"], label_config["list"], left, right, shuffle)
        # {list:[]}
        elif "list" in label_config.keys():
            label_lists = generate_label_lists(label_config["list"], left, right, shuffle)
        # {[],[],[]}
        else:
            label_lists = dict_to_list(label_config)
    # 生成data lists
    # {}
    if len(data_config) == 0:
        size = dataset.train_data_size // clients
        data_lists = generate_data_lists(size, size, clients, label_lists)
    # max,min
    else:
        data_lists = generate_data_lists(data_config["max"], data_config["min"], clients, label_lists)
    # 生成datasets
    dataset.datasets = utils.Tools.generate_non_iid_data(dataset.train_data, dataset.train_labels, label_lists, data_lists)
    # 保存label至配置文件
    dataset.iid_config['label'] = list_to_dict(label_lists)
