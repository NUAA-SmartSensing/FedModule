import copy

import torch

from queuemanager.SingleQueueManager import SingleQueueManager
from utils.Tools import to_dev, to_cpu


class VCM_Factory:
    @staticmethod
    def create_VCM(config, existing_version_model):
        beta = config.get('beta')
        gamma_vcm = config.get('gamma_vcm')
        kwargs = {k: v for k, v in {'beta': beta, 'gamma_vcm': gamma_vcm}.items() if v is not None}
        if config['update'] == 'direct':
            return VCM_Direct(existing_version_model, **kwargs)
        elif config['update'] == 'first':
            return VCM_First(existing_version_model, **kwargs)
        elif config['update'] == 'second':
            return VCM_Second(existing_version_model, **kwargs)
        else:
            return VCM_None(existing_version_model, **kwargs)


class VCMQueueManager(SingleQueueManager):
    def __init__(self, config):
        super().__init__(config)
        self.time_window = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.existing_versions = {}
        self.existing_versions_model = {}
        self.current_t = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        self.vcm = VCM_Factory.create_VCM(config['vcm'], self.existing_versions_model)
        self.client_num = 0
        self.latest_model = {}

    def receive(self, nums, *args, **kwargs):
        self.receiver.receive(self, nums, *args, **kwargs)

    def put(self, update, *args, **kwargs):
        if update['time_stamp'] != self.current_t.get_time() and update['time_stamp'] != self.schedule_t.get_time() - 1:
            self.vcm.correct(self.latest_model, update)
        else:
            self.client_num += 1
        super().put(update, *args, **kwargs)
        self.existing_versions[update['time_stamp']] -= 1
        if self.existing_versions[update['time_stamp']] == 0:
            self.existing_versions.pop(update['time_stamp'])
            self.existing_versions_model.pop(update['time_stamp'])

    def set_time_window(self, time_window):
        self.time_window = time_window
        self.client_num = 0

    def set_version(self, version_id, version_num, version_model):
        version_model = to_dev(copy.deepcopy(version_model), self.device)
        if version_id == 1:
            self.vcm.init(version_model)
        self.latest_model = version_model
        self.existing_versions[version_id] = version_num
        self.existing_versions_model[version_id] = version_model


class VCM_None:
    def __init__(self, exising_version_model, beta=0, gamma_vcm=0):
        pass

    def correct(self, latest_model, update_dict):
        pass

    def t1_add(self, version_model):
        pass

    def init(self, version_model):
        pass


class VCM_Direct:
    def __init__(self, exising_version_model, beta=0.9, gamma_vcm=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.gamma_vcm = gamma_vcm
        self.t1 = 0
        self.existing_versions_model = exising_version_model

    def correct(self, latest_model, update_dict):
        self.correct_update(latest_model, update_dict)
        self.correct_data_sum(update_dict)

    def correct_update(self, latest_model, update_dict):
        update_dict["weights"] = to_dev(update_dict["weights"], self.device)
        for key, var in update_dict["weights"].items():
            update_dict["weights"][key] = var + self.gamma_vcm * (
                    latest_model[key] -
                    self.existing_versions_model[update_dict["time_stamp"]][key])
        update_dict["weights"] = to_cpu(update_dict["weights"])

    def correct_data_sum(self, update_dict):
        pass

    def t1_add(self, version_model):
        self.t1 += 1

    def init(self, version_model):
        pass


class VCM_First(VCM_Direct):
    def __init__(self, exising_version_model, beta=0.01, gamma_vcm=1.0):
        super().__init__(exising_version_model, beta, gamma_vcm)
        self.exp_avgs = {}
        self.global_exp_avg = {}
        self.t2 = 0

    def init(self, version_model):
        for k, v in version_model.items():
            self.global_exp_avg[k] = torch.zeros_like(v, memory_format=torch.preserve_format, dtype=torch.float32).to(
                self.device)

    def correct_update(self, latest_model, update_dict):
        update_dict["weights"] = to_dev(update_dict["weights"], self.device)
        if update_dict["client_id"] not in self.exp_avgs:
            self.exp_avgs[update_dict["client_id"]] = {}
            for k, v in update_dict["weights"].items():
                self.exp_avgs[update_dict["client_id"]][k] = torch.zeros_like(v, memory_format=torch.preserve_format,
                                                                              dtype=torch.float32).to(self.device)
        self.t2 += 1
        bias_correction1 = 1 - self.beta ** self.t1
        bias_correction2 = 1 - self.beta ** self.t2
        for key, var in update_dict["weights"].items():
            self.exp_avgs[update_dict["client_id"]][key].mul_(self.beta).add_(var, alpha=1 - self.beta)
            update_dict["weights"][key] = update_dict["weights"][key] + self.gamma_vcm * abs(
                self.exp_avgs[update_dict["client_id"]][key] * bias_correction1 / (
                        bias_correction2 * self.global_exp_avg[key] + 1e-8)) * (
                                                      latest_model[key] -
                                                      self.existing_versions_model[
                                                          update_dict["time_stamp"]][
                                                          key])
        update_dict["weights"] = to_cpu(update_dict["weights"])

    def t1_add(self, version_model):
        super().t1_add(version_model)
        for key, var in version_model.items():
            var = var.to(self.device)
            self.global_exp_avg[key].mul_(self.beta).add_(var, alpha=1 - self.beta)


class VCM_Second(VCM_First):
    def __init__(self, exising_version_model, beta=0.9, gamma_vcm=0.9):
        super().__init__(exising_version_model, beta, gamma_vcm)

    def correct_update(self, latest_model, update_dict):
        update_dict["weights"] = to_dev(update_dict["weights"], self.device)
        if update_dict["client_id"] not in self.exp_avgs:
            self.exp_avgs[update_dict["client_id"]] = {}
            for k, v in update_dict["weights"].items():
                self.exp_avgs[update_dict["client_id"]][k] = torch.zeros_like(v, memory_format=torch.preserve_format,
                                                                              dtype=torch.float32).to(self.device)
        self.t2 += 1
        bias_correction1 = 1 - self.beta ** self.t1
        bias_correction2 = 1 - self.beta ** self.t2
        for key, var in update_dict["weights"].items():
            self.exp_avgs[update_dict["client_id"]][key].mul_(self.beta).addcmul_(var, var.conj(), value=1 - self.beta)
            update_dict["weights"][key] = update_dict["weights"][key] + self.gamma_vcm * (
                        self.exp_avgs[update_dict["client_id"]][key] * bias_correction1 / (
                            bias_correction2 * self.global_exp_avg[key] + 1e-8)).sqrt() * (
                                                      latest_model[key] -
                                                      self.existing_versions_model[update_dict["time_stamp"]][key])
        update_dict["weights"] = to_cpu(update_dict["weights"])

    def t1_add(self, version_model):
        super().t1_add(version_model)
        for key, var in version_model.items():
            var = var.to(self.device)
            self.global_exp_avg[key].mul_(self.beta).addcmul_(var, var.conj(), value=1 - self.beta)
