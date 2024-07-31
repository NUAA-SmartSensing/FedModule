import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from utils.DatasetUtils import FLDataset
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import saveAns


class TestEachClass:
    def __init__(self):
        self.results = []

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = len(set(self.test_data.targets.data.numpy()))
        class_accuracies = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for data in dl:
                inputs, labels = data
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = self.server_network(inputs)
                _, id = torch.max(outputs.data, 1)
                test_loss += self.loss_func(outputs, labels).detach().item()
                test_correct += torch.sum(id == labels.data).cpu().numpy()
                c = (id == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_accuracies[label] += c[i].item()
                    class_total[label] += 1
            accuracy = test_correct / len(dl)
            loss = test_loss / len(dl)
            self.loss_list.append(loss)
            self.accuracy_list.append(accuracy)
            epoch_result = {}
            print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)
            for i in range(num_classes):
                acc = class_accuracies[i] / class_total[i]
                epoch_result[i] = acc
                print(f"acc on class {i}: {acc:.4f}")
            self.results.append(epoch_result)
            if self.config['enabled']:
                wandb.log({'accuracy': accuracy, 'loss': loss, 'class_acc': epoch_result})
        return accuracy, loss

    def __del__(self):
        global_var = GlobalVarGetter.get()
        saveAns(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/",
                             global_var['global_config']["experiment"],
                             "class_acc.txt"), self.results)
        if self.config['enabled']:
            wandb.save(os.path.join(wandb.run.dir, "class_acc.txt"), self.results)


class TestMultiTask:
    def __init__(self):
        global_var = GlobalVarGetter().get()
        self.test_index_list = global_var["test_index_list"]
        self.label_mapping = global_var["label_mapping"] if "label_mapping" in global_var else None

    def run_server_test(self, epoch):
        for task, task_list in enumerate(self.test_index_list):
            self.run_sub_test(task, epoch, task_list)

    def run_sub_test(self, task, epoch, index_list):
        dl = DataLoader(FLDataset(self.test_data, index_list), batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        classes = set(self.test_data.targets.data.numpy())
        if self.label_mapping is not None:
            num_classes = len(set(self.label_mapping[i] for i in classes))
        else:
            num_classes = len(set(self.test_data.targets.data.numpy()))
        class_accuracies = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for inputs, labels in dl:
                if self.label_mapping is not None:
                    labels = torch.LongTensor([self.label_mapping[i.item()] for i in labels])
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = self.server_network(inputs)
                _, id = torch.max(outputs.data, 1)
                test_loss += self.loss_func(outputs, labels).detach().item()
                test_correct += torch.sum(id == labels.data).cpu().numpy()
                c = (id == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_accuracies[label] += c[i].item()
                    class_total[label] += 1
            accuracy = test_correct / len(dl)
            loss = test_loss / len(dl)
            self.loss_list.append(loss)
            self.accuracy_list.append(accuracy)
            print(f'Epoch(t): {epoch}-{task} accuracy: {accuracy} {loss}')
            for i in range(num_classes):
                print(f"acc on class {i}: {class_accuracies[i] / class_total[i]:.4f}")
            if self.config['enabled']:
                wandb.log({'task': task, 'accuracy': accuracy, 'loss': loss})
        return accuracy, loss
