import os.path

import numpy as np
import torch
import wandb

from core.handlers.Handler import Handler
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import saveAns


class ClientTestHandler(Handler):
    def __init__(self):
        super().__init__()
        global_var = GlobalVarGetter.get()
        self.test_every = global_var['config']['client'].get('test_every', 1)

    def _handle(self, request):
        client = request.get('client')
        epoch = request.get('epoch')
        if epoch % self.test_every != 0:
            return request
        config = client.config
        if 'test' in config:
            test_func = ModuleFindTool.find_class_by_path(config['test'])
        elif hasattr(client, 'test'):
            request['test_res'] = client.test()
            return request
        else:
            test_func = BasicTest
        request['test_res'] = test_func(client.test_dl, client.model, client.loss_func, client.dev, epoch, client)
        return request


class ClientPostTestHandler(Handler):
    def __init__(self):
        super().__init__()
        global_var = GlobalVarGetter.get()
        self.cloud_enabled = global_var['config']['wandb']['enabled']
        self.file_enabled = global_var['config']['global']['save']
        self.accuracy_list = []
        self.loss_list = []

    def _handle(self, request):
        if 'test_res' not in request:
            return request
        acc, loss = request.get('test_res')
        epoch = request.get('epoch')
        client = request.get('client')
        client.upload_item('accuracy', acc)
        client.upload_item('loss', loss)
        print('Client', client.client_id, 'tested, accuracy:', acc, 'loss', loss)
        if self.cloud_enabled:
            wandb.log({'client_id': client.client_id, 'accuracy': acc, 'loss': loss}, step=epoch)
        self.accuracy_list.append(acc)
        self.loss_list.append(loss)
        return request

    def run_once(self, request):
        if self.file_enabled:
            client = request.get('client')
            experiment = request.get('global_var')['config']['global']['experiment']
            client_id = client.client_id
            path1 = os.path.join('../results', experiment, f'{client_id}_accuracy.txt')
            path2 = os.path.join('../results', experiment, f'{client_id}_loss.txt')
            client.add_final_callback(saveAns, path1, self.accuracy_list)
            client.add_final_callback(saveAns, path2, self.loss_list)


class ServerTestHandler(Handler):
    def _handle(self, request):
        updater = request.get('updater')
        epoch = request.get('epoch')
        config = updater.config
        if 'test' in config:
            test_func = ModuleFindTool.find_class_by_path(config['test'])
        elif hasattr(updater, 'test'):
            request['test_res'] = updater.test()
            return request
        else:
            test_func = BasicTest
        request['test_res'] = test_func(updater.test_dl, updater.model, updater.loss_func, updater.dev, epoch, updater)
        return request


class ServerPostTestHandler(Handler):
    def __init__(self):
        super().__init__()
        global_var = GlobalVarGetter.get()
        self.cloud_enabled = global_var['config']['wandb']['enabled']
        self.file_enabled = global_var['config']['global']['save']
        self.accuracy_list = []
        self.loss_list = []

    def _handle(self, request):
        if 'test_res' not in request:
            return request
        acc, loss = request.get('test_res')
        epoch = request.get('epoch')

        # 处理并记录精度
        if isinstance(acc, dict):
            # 获取总精度用于打印和本地保存
            total_acc = acc.get('total', acc.get('avg', list(acc.values())[0] if acc else 0))
            print('Epoch', epoch, 'tested, accuracy:', total_acc, 'loss:', loss)

            # 上传到wandb，每个精度指标单独上传
            if self.cloud_enabled:
                log_data = {}
                # 添加所有精度指标
                for key, value in acc.items():
                    log_data[f'accuracy/{key}'] = value

                # 处理损失值
                if isinstance(loss, dict):
                    for key, value in loss.items():
                        log_data[f'loss/{key}'] = value
                else:
                    log_data['loss'] = loss

                # 上传所有指标
                wandb.log(log_data, step=epoch)
        else:
            # 兼容处理非字典格式的精度结果
            print('Epoch', epoch, 'tested, accuracy:', acc, 'loss:', loss)
            if self.cloud_enabled:
                wandb.log({'accuracy/total': acc, 'loss': loss}, step=epoch)
        self.accuracy_list.append(acc)
        self.loss_list.append(loss)

        return request

    def run_once(self, request):
        if self.file_enabled:
            updater = request.get('updater')
            experiment = request.get('global_var')['config']['global']['experiment']
            path1 = os.path.join('../results', experiment, f'accuracy.txt')
            path2 = os.path.join('../results', experiment, f'loss.txt')
            updater.add_final_callback(saveAns, path1, self.accuracy_list)
            updater.add_final_callback(saveAns, path2, self.loss_list)


def BasicTest(test_dl, model, loss_func, dev, epoch, obj=None):
    test_correct = 0
    test_loss = 0
    total_samples = 0  # 添加样本计数
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_loss += loss_func(outputs, labels).detach().item()
            total_samples += labels.size(0)  # 累加每个批次的实际样本数
    accuracy = (test_correct * 100) / total_samples
    loss = test_loss / len(test_dl)
    # 创建精度字典，与其他测试函数保持一致的格式
    accuracy_dict = {'total': float(accuracy)}
    return accuracy_dict, loss


def TestEachClass(test_dl, model, loss_func, dev, epoch, obj=None):
    test_correct = 0
    test_loss = 0
    total_samples = 0  # 添加样本计数
    num_classes = len(set(obj.test_ds.targets.data.numpy()))
    class_accuracies = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_loss += loss_func(outputs, labels).detach().item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            c = (id == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_accuracies[label] += c[i].item()
                class_total[label] += 1
            total_samples += labels.size(0)  # 累加每个批次的实际样本数
        accuracy = test_correct / total_samples  # 使用总样本数
        loss = test_loss / len(test_dl)

        # 创建包含总精度和各类别精度的字典
        accuracy_dict = {'total': float(accuracy) * 100}
        for i in range(num_classes):
            acc = class_accuracies[i] / class_total[i]
            accuracy_dict[f'class_{i}'] = float(acc) * 100
            print(f"acc on class {i}: {acc * 100:.2f}")

    return accuracy_dict, loss


def TestMultiTask(test_dl, model, loss_func, dev, epoch, obj=None):
    avg_acc = 0
    avg_loss = 0

    # 创建包含平均精度和各任务精度的字典
    accuracy_dict = {}
    loss_dict = {}

    for task, task_list in enumerate(obj.test_index_list):
        acc, loss = _sub_test_for_multi_task(test_dl, model, loss_func, dev, epoch, task, obj)
        accuracy_dict[f'task_{task}'] = float(acc)
        loss_dict[f'task_{task}'] = float(loss)
        avg_acc += acc / len(obj.test_index_list)
        avg_loss += loss / len(obj.test_index_list)

    accuracy_dict['avg'] = float(avg_acc)
    loss_dict['avg'] = float(avg_loss)

    return accuracy_dict, loss_dict


def _sub_test_for_multi_task(test_dl, model, loss_func, dev, epoch, task, obj=None):
    test_correct = 0
    test_loss = 0
    total_samples = 0  # 添加样本计数
    classes = set(obj.test_ds.targets.data.numpy())
    if obj.label_mapping is not None:
        num_classes = len(set(obj.label_mapping[i] for i in classes))
    else:
        num_classes = len(set(obj.test_ds.targets.data.numpy()))
    class_accuracies = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    with torch.no_grad():
        for inputs, labels in test_dl:
            if obj.label_mapping is not None:
                labels = torch.LongTensor([obj.label_mapping[i.item()] for i in labels])
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_loss += loss_func(outputs, labels).detach().item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            c = (id == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_accuracies[label] += c[i].item()
                class_total[label] += 1
            total_samples += labels.size(0)  # 累加每个批次的实际样本数
        accuracy = test_correct / total_samples  # 使用总样本数
        loss = test_loss / len(test_dl)
        print(f'Epoch(t): {epoch}-{task} accuracy: {accuracy * 100:.2f}{loss}')
        for i in range(num_classes):
            print(f"acc on class {i}: {class_accuracies[i] / class_total[i] * 100:.2f}")
    return accuracy * 100, loss
