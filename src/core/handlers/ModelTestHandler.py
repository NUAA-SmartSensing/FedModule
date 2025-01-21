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
        print('Epoch', epoch, 'tested, accuracy:', acc, 'loss', loss)
        if self.cloud_enabled:
            wandb.log({'accuracy': acc, 'loss': loss}, step=epoch)
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
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_loss += loss_func(outputs, labels).detach().item()
    accuracy = (test_correct * 100) / (len(test_dl) * test_dl.batch_size)
    loss = test_loss / len(test_dl)
    return accuracy, loss


def TestEachClass(test_dl, model, loss_func, dev, epoch, obj=None):
    test_correct = 0
    test_loss = 0
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
        accuracy = test_correct / len(test_dl)
        loss = test_loss / len(test_dl)
        detail_acc = {}
        for i in range(num_classes):
            acc = class_accuracies[i] / class_total[i]
            detail_acc[i] = float(acc)
            print(f"acc on class {i}: {acc:.4f}")
    return [float(accuracy), detail_acc], loss


def TestMultiTask(test_dl, model, loss_func, dev, epoch, obj=None):
    avg_acc, avg_loss = 0, 0
    total_acc = [avg_acc]
    total_loss = [avg_loss]

    for task, task_list in enumerate(obj.test_index_list):
        acc, loss = _sub_test_for_multi_task(test_dl, model, loss_func, dev, epoch, task, obj)
        total_acc.append(float(acc))
        total_loss.append(float(loss))
        avg_acc += acc / len(obj.test_index_list)
        avg_loss += loss / len(obj.test_index_list)
    return total_acc, total_loss


def _sub_test_for_multi_task(test_dl, model, loss_func, dev, epoch, task, obj=None):
    test_correct = 0
    test_loss = 0
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
        accuracy = test_correct / len(test_dl)
        loss = test_loss / len(test_dl)
        print(f'Epoch(t): {epoch}-{task} accuracy: {accuracy} {loss}')
        for i in range(num_classes):
            print(f"acc on class {i}: {class_accuracies[i] / class_total[i]:.4f}")
    return accuracy, loss

