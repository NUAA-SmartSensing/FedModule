import copy

import torch.nn.functional as F
from sklearn.cluster import KMeans

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import to_cpu


class FedDL(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()
        self.client_weights = {'global': copy.deepcopy(self.global_var['global_model'].state_dict())}
        self.global_var['scheduler'].server_weights = self.client_weights
        self.clusterer = KMeans(n_clusters=self.config['n_clusters'], n_init="auto", random_state=0)
        self.updater_thread = None

    def update_server_weights(self, epoch, update_list):
        self.updater_thread = self.global_var['updater']
        self.client_weights = {'global': copy.deepcopy(self.updater_thread.model.state_dict())}
        cluster_group = {}
        # 计算kld
        clusters = {0: [k["client_id"] for k in update_list]}
        id_update_idx_map = {}
        for i in range(len(update_list)):
            id_update_idx_map[update_list[i]["client_id"]] = i
        for key in update_list[0]["weights"].keys():
            clusters = self.kld_cluster(key, update_list, clusters, id_update_idx_map)
            cluster_group[key] = clusters
        # 更新
        for key, clusters in cluster_group.items():
            for _, cluster in clusters.items():
                updated_parameter = None
                data_sum = 0
                for i in cluster:
                    data_sum += update_list[id_update_idx_map[i]]["data_sum"]
                    if updated_parameter is None:
                        updated_parameter = update_list[id_update_idx_map[i]]["weights"][key] * update_list[id_update_idx_map[i]]["data_sum"]
                    else:
                        updated_parameter += update_list[id_update_idx_map[i]]["weights"][key] * update_list[id_update_idx_map[i]]["data_sum"]

                updated_parameter = updated_parameter / data_sum
                for i in cluster:
                    if i not in self.client_weights.keys():
                        self.client_weights[i] = {}
                    self.client_weights[i][key] = updated_parameter.clone()

        return self.updater_thread.model.state_dict(), self.client_weights

    def kld_cluster(self, key, update_list, clusters: dict, id_update_idx_map):
        label = 0
        class_num = {}
        new_clusters_dict = {}
        for k in clusters.keys():
            class_num[k] = len(clusters[k])
        for cls in class_num.keys():
            if class_num[cls] < self.config['n_clusters']:
                new_clusters_dict[label] = clusters[cls]
                label += 1
            else:
                kld = [[0 for _ in range(class_num[cls])] for _ in range(class_num[cls])]
                index = 0
                id_idx_map = {}
                idx_id_map = {}
                for i in clusters[cls]:
                    id_idx_map[i] = index
                    idx_id_map[index] = i
                    index += 1
                for i in range(class_num[cls]):
                    for j in range(class_num[cls]):
                        if i < j:
                            kld[i][j] = F.kl_div(update_list[id_update_idx_map[idx_id_map[i]]]["weights"][key].softmax(dim=-1).log(), update_list[id_update_idx_map[idx_id_map[j]]]["weights"][key].softmax(dim=-1),
                                                 reduction='batchmean')
                            kld[j][i] = kld[i][j]
                        else:
                            kld[i][i] = F.kl_div(update_list[id_update_idx_map[idx_id_map[i]]]["weights"][key].softmax(dim=-1).log(), update_list[id_update_idx_map[idx_id_map[i]]]["weights"][key].softmax(dim=-1),
                                                 reduction='batchmean')
                # 聚类
                new_clusters = self.clusterer.fit_predict(to_cpu(kld))
                # 更新
                for k in range(self.config['n_clusters']):
                    for i in range(len(new_clusters)):
                        if new_clusters[i] == k:
                            if label in new_clusters_dict.keys():
                                new_clusters_dict[label].append(idx_id_map[i])
                            else:
                                new_clusters_dict[label] = [idx_id_map[i]]
                    label += 1
        return new_clusters_dict
