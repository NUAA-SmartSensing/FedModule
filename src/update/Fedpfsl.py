import copy
import torch
import numpy as np

from sklearn.cluster import KMeans
from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class Fedpfsl(AbstractUpdate):
    def __init__(self, config):
        self.global_var = GlobalVarGetter().get()
        self.client_weights = {'global': copy.deepcopy(self.global_var['server_network'].state_dict())}
        self.global_var['scheduler'].server_weights = self.client_weights
        self.config = config
        self.prune_ratios = {}
        self.clusterer = KMeans(n_clusters=self.config['n_clusters'], n_init="auto", random_state=0)
        self.updater_thread = None

    def update_server_weights(self, epoch, update_list):
        self.updater_thread = self.global_var['updater']
        self.client_weights = {'global': copy.deepcopy(self.updater_thread.server_network.state_dict())}
        # 聚类
        clusters = self.cos_similarity_cluster(update_list)
        # 统计客户端的剪枝率
        for i in range(len(update_list)):
            if update_list[i]["client_id"] in self.prune_ratios.keys():
                continue
            else:
                self.prune_ratios[update_list[i]["client_id"]] = update_list[i]["prune_ratio"]

        # 按组聚合
        for _, cluster in clusters.items():
            updated_parameters = {}
            data_sum = 0
            for key, var in update_list[0]["weights"].items():
                updated_parameters[key] = None

            for i in cluster:
                data_sum += update_list[i]["data_sum"]

            for i in cluster:
                for key, var in update_list[i]["weights"].items():
                    if updated_parameters[key] is None:
                        updated_parameters[key] = update_list[i]["weights"][key] * (update_list[i]["data_sum"]
                                                                                    * (1 - self.prune_ratios[update_list[i]["client_id"]]) / data_sum)
                    else:
                        updated_parameters[key] += update_list[i]["weights"][key] * (update_list[i]["data_sum"]
                                                                                    * (1 - self.prune_ratios[update_list[i]["client_id"]]) / data_sum)

            # 将聚合的模型给到对应的客户端
            for i in cluster:
                if update_list[i]["client_id"] not in self.client_weights.keys():
                    self.client_weights[update_list[i]["client_id"]] = {}
                for key, var in update_list[0]["weights"].items():
                    self.client_weights[update_list[i]["client_id"]][key] = updated_parameters[key] / (1 - self.prune_ratios[update_list[i]["client_id"]])

        return self.updater_thread.server_network.state_dict(), self.client_weights

    # 根据余弦相似度进行聚类
    def cos_similarity_cluster(self, update_list):
        label = 0
        # 余弦相似矩阵
        weight_list = []
        new_clusters_dict = {f'array{i}': [] for i in range(self.config['n_clusters'])}
        for i in range(len(update_list)):
            weight_list.append(torch.cat([p.view(-1) for _, p in update_list[i]["weights"].items()]))
        cos_similarity = [[0 for _ in range(len(update_list))] for _ in range(len(update_list))]
        for i in range(len(update_list)):
            for j in range(len(update_list)):
                if i < j:
                    cos_similarity[i][j] = torch.dot(weight_list[i], weight_list[j]) / np.sqrt(
                        torch.dot(weight_list[i], weight_list[i]) * torch.dot(weight_list[j], weight_list[j]))
                    cos_similarity[j][i] = cos_similarity[i][j]
                else:
                    cos_similarity[i][i] = 1
        # 聚类
        new_clusters = self.clusterer.fit_predict(cos_similarity)
        for k in range(self.config['n_clusters']):
            for i in range(len(new_clusters)):
                if new_clusters[i] == k + 1:
                    key = f'array{k}'
                    new_clusters_dict[key].append(i)

        return new_clusters_dict
