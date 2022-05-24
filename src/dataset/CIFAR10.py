import copy

import torchvision
from torchvision import transforms


class CIFAR10:
    def __init__(self, clients, is_iid=False):
        # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        # test_transform = transforms.Compose([transforms.ToTensor()])
        # 获取数据集
        # train_datasets = datasets.CIFAR10(root='../data/', train=True,
        #                                   transform=train_transform, download=True)
        # test_datasets = datasets.CIFAR10(root='../data/', train=False,
        #                                  transform=test_transform, download=True)
        # self.train_data = train_datasets.data
        # self.train_labels = np.array(train_datasets.targets)
        # test_data = test_datasets.data
        # self.test_datasets = test_datasets
        #
        # self.train_data_size = self.train_data.shape[0]
        # self.datasets = []
        # self.train_images = self.train_data.reshape(self.train_data.shape[0],
        #                                   self.train_data.shape[1] * self.train_data.shape[2] * self.train_data.shape[2])
        # self.train_images = self.train_images.astype(np.float32)
        # self.train_images = np.multiply(self.train_images, 1.0/ 255.0)
        # if is_iid:
        #     order = np.arange(self.train_data_size)
        #     np.random.shuffle(order)
        #     self.train_data = self.train_images[order]
        #     self.train_labels = np.array(self.train_labels)[order]
        # else:
        #     print("generating...")
        #     order = np.argsort(self.train_labels)
        #     # saveOrder("IID/MNIST/order.txt", list(order))
        #     # order = get_order_as_tuple("../results/IID/MNIST/order.txt")
        #     self.train_data = self.train_images[order]
        #     self.train_labels = np.array(self.train_labels)[order]
        # total_clients = clients
        # clients = clients // 2
        # shard_size = self.train_data_size // clients // 4
        # for i in range(clients):
        #     client_data1 = self.train_data[shard_size * i: shard_size * (i + 1)]
        #     client_data2 = self.train_data[
        #                    shard_size * clients + shard_size * i: shard_size * clients + shard_size * (i + 1)]
        #     client_data3 = self.train_data[
        #                    shard_size * clients * 2 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_data4 = self.train_data[
        #                    shard_size * clients * 2 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_data5 = self.train_data[
        #                    shard_size * clients * 4 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_label1 = self.train_labels[shard_size * i: shard_size * (i + 1)]
        #     client_label2 = self.train_labels[
        #                     shard_size * clients + shard_size * i: shard_size * clients + shard_size * (i + 1)]
        #     client_label3 = self.train_labels[
        #                     shard_size * clients * 2 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_label4 = self.train_labels[
        #                     shard_size * clients * 2 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_label5 = self.train_labels[
        #                     shard_size * clients * 4 + shard_size * i: shard_size * clients * 2 + shard_size * (i + 1)]
        #     client_data, client_label = np.vstack(
        #         (client_data1, client_data2, client_data3, client_data4, client_data5)), np.hstack(
        #         (client_label1, client_label2, client_label3, client_label4, client_label5))
        #     self.datasets.append(TensorDataset(torch.tensor(client_data), torch.tensor(client_label)))
        # for i in range(total_clients - clients):
        #     self.datasets.append(copy.deepcopy(self.datasets[i]))
        self.datasets = []
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_datasets = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        self.test_datasets = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        for i in range(clients):
            self.datasets.append(copy.deepcopy(train_datasets))

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets


if __name__ == '__main__':
    CIFAR10(50)