import numpy as np
import torch
from sklearn.cluster import KMeans

from group.AbstractGroup import AbstractGroup
from utils.GlobalVarGetter import GlobalVarGetter


class CosMatrix(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.n = config["n"]
        self.training_params = GlobalVarGetter.get()['training_params']

    def group(self, client_list, latency_list, *args, **kwargs):
        if len(args) == 0:
            return [client_list], 1
        weight_list = args[0]
        weight_list = [update['weights'] for update in weight_list]
        n = len(weight_list)
        for i in range(n):
            weight_list[i] = torch.cat([p.view(-1) for _, p in weight_list[i].items()])
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = cos_sim(weight_list[i], weight_list[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        X = sim_matrix.reshape(-1, 1)
        # KMeans聚类
        kmeans = KMeans(n_clusters=self.n)
        kmeans.fit(X)
        labels = kmeans.labels_
        col_labels = labels % sim_matrix.shape[1]
        groups = [[] for i in range(max(labels) + 1)]
        for i in range(len(weight_list)):
            groups[col_labels[i]].append(i)
        return groups, len(groups)

    def check_update(self):
        return

    def compute_cos_sim(self, a, b):
        cos_sims = []
        for k in a.keys():
            if self.training_params[k]:
                cos_sims.append(cos_sim(a[k], b[k]))
        return sum(cos_sims) / len(cos_sims)


def cos_sim(a, b):
    return (a * b).sum() / (a.norm() * b.norm())
