import json

import wandb
import matplotlib.pyplot as plt

import numpy as np

from utils.Tools import getJson


def smooth(data, window_size):
    smoothed = []
    for i in range(len(data)):
        left = max(0, i - window_size)
        right = min(len(data) - 1, i + window_size)
        smoothed.append(np.mean(data[left:right + 1]))
    return smoothed


def write_plot(save=False, save_path="example.pdf"):
    api = wandb.Api()
    twafl = api.run("/team-zhao/CIFAR10_noniid0.5/runs/d986ln6q").history()
    fedadam = api.run("/team-zhao/CIFAR10_noniid0.5/runs/3wp7e3zc").history()
    fedavg = api.run("/team-zhao/CIFAR10_noniid0.5/runs/01knzlbc").history()

    # plt.title("CIFAR10 + non-i.i.d0.5", fontsize='17')
    plt.plot(twafl[:]["_runtime"]/60, smooth(twafl[:]["accuracy"], 5), '-', label="TWAFL", markersize=5, markevery=5)
    plt.plot(fedadam[:]["_runtime"]/60, smooth(fedadam[:]["accuracy"], 5), '-o', label="FedAdam", markersize=5, markevery=5)
    plt.plot(fedavg[:]["_runtime"]/60, smooth(fedavg[:]["accuracy"], 5), '-3', label="FedAvg", markersize=5, markevery=5)
    plt.xticks(range(0, 150, 50))  # 设置横坐标刻度为给定的年份
    plt.xlabel('Time(m)', fontsize='17')  # 设置横坐标轴标题
    plt.yticks(range(20, 80, 10))
    plt.ylim(ymin=20)
    plt.xlim(xmax=150)
    plt.ylabel('Accuracy', fontsize='17')
    plt.legend()  # 显示图例，即每条线对应 label 中的内容
    plt.grid()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


def write_data_distri(data, save=False, save_path="sample.pdf"):
    # 从提供的数据中提取客户端编号、样本总量和每个标签的样本量
    if isinstance(data, list):
        label_counts = data
    elif isinstance(data, str):
        if not data.endswith("json"):
            raise Exception("仅支持json类型")
        label_counts = getJson(data)
    else:
        raise Exception("未支持的类型")


    # 设置标签
    labels = [f"Client {client_id + 1}" for client_id in range(len(label_counts))]

    # 提取每个标签的样本量
    label_samples = np.array([[v for _,v in label_sample.items()] for label_sample in label_counts])

    # 绘制条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    bar_positions = np.arange(len(label_counts))
    bottom = np.zeros(len(label_counts))
    for i in range(len(label_samples)):
        ax.bar(bar_positions - bar_width / 2, label_samples[:, i], bar_width, bottom, label=f"Label {i}")
        bottom += label_samples[:, i]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Sample Count')

    # ax.set_title('Sample Distribution Across Clients')
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
