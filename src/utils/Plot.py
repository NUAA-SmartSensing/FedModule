import wandb
import matplotlib.pyplot as plt

import numpy as np


def smooth(data, window_size):
    smoothed = []
    for i in range(len(data)):
        left = max(0, i - window_size)
        right = min(len(data) - 1, i + window_size)
        smoothed.append(np.mean(data[left:right + 1]))
    return smoothed


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
plt.show()
# plt.savefig("example.pdf")
