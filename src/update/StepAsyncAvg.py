from fedasync import UpdaterThread
import copy
import torch


class StepAsyncAvg:
    def __init__(self, config, updater_thread):
        self.config = config
        self.num_cnt = {}
        self.sum_cnt = 0
        self.updater_thread = updater_thread

    def update_server_weights(self, epoch, update_list):
        rho = self.config["rho"]
        self.sum_cnt += len(update_list)
        alpha = self.config["alpha"]
        server_weights = copy.deepcopy(self.updater_thread.server_network.state_dict())

        for i in range(len(update_list)):
            for key, var in update_list[i]["weights"].items():
                if torch.cuda.is_available():
                    update_list[i]["weights"][key] = update_list[i]["weights"][key].cuda()

        # 求每个客户端的上传频率q
        q = []
        for i in range(len(update_list)):
            if update_list[i]["client_id"] in self.num_cnt.keys():
                self.num_cnt[update_list[i]["client_id"]] += 1
            else:
                self.num_cnt[update_list[i]["client_id"]] = 1
            q.append(self.num_cnt[update_list[i]["client_id"]] / self.sum_cnt)

        # 求每个模型与全局模型的偏差λ
        w_global = torch.cat([p.view(-1) for _, p in server_weights.items()])
        w_global_norm2 = torch.dot(w_global, w_global)
        lambda_list = []
        for i in range(len(update_list)):
            lambda_list.append(
                torch.dot(w_global, torch.cat([p.view(-1) for _, p in update_list[i]["weights"].items()]))
                / w_global_norm2 - 1)

        # 求聚合系数和分母
        aggregation_factor = []
        aggregation_sum = 0
        for i in range(len(update_list)):
            aggregation_factor.append(update_list[i]["data_sum"] * torch.exp(-rho * torch.abs(lambda_list[i] / q[i])))
            aggregation_sum += aggregation_factor[i]
        for i in range(len(update_list)):
            aggregation_factor[i] = aggregation_factor[i] / aggregation_sum


        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * aggregation_factor[0]
        for i in range(len(update_list) - 1):
            for key, var in update_list[i + 1]["weights"].items():
                updated_parameters[key] += update_list[i + 1]["weights"][key] * aggregation_factor[i + 1]

        # 异步更新
        for key, var in server_weights.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
            updated_parameters[key] = (alpha * updated_parameters[key] + (1 - alpha) * server_weights[key])

        # 下发给客户端的权重
        self.updater_thread.global_var['scheduler'].server_weights = updated_parameters
        return updated_parameters
