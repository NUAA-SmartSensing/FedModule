# 联邦学习简易框架

>keywords: `federated-learning`, `asynchronous`

## 初衷

本项目的初衷是我本科毕设期间需要完成搭建一个异步联邦学习框架，并且在其之上完成一些实验。

可当我去github尝试搜索项目时，发现异步联邦学习闭源之深，几乎没有开源项目。并且主流框架也基本不兼容异步，只支持同步FL。因此促生了该项目的诞生。

## 基本配置

python3.8 + pytorch + macos

在linux进行过验证

支持单GPU，尚未进行多GPU优化

## 项目目录

```text
.
├── config.json                               配置文件                                
├── readme.md                                 
├── requirements.txt
└── src
    ├── data                                  数据集下载位置
    ├── dataset                               数据集类
    │   ├── CIFAR10.py
    │   ├── MNIST.py
    │   └── __init__.py
    ├── fedasync                              异步联邦学习
    │   ├── AsyncClient.py              			客户端类
    │   ├── AsyncClientManager.py       			客户端管理类
    │   ├── AsyncServer.py              			服务器类
    │   ├── CheckInThread.py            			CheckIn进程
    │   ├── Queue.py
    │   ├── SchedulerThread.py          			调度进程
    │   ├── Time.py
    │   ├── UpdaterThread.py            			聚合进程
    │   ├── __init__.py
    │   └── main.py
    ├── fedsync                               同步联邦学习
    │   └── __init__.py
    ├── model                                 模型类
    │   ├── CNN.py
    │   ├── ConvNet.py
    │   └── __init__.py
    ├── results                               实验结果
    ├── schedule															调度算法类
    │   ├── RandomSchedule.py
    │   └── __init__.py
    ├── test																	测试用
    ├── update																聚合算法类
    │   ├── Avg.py
    │   ├── FedAsync.py
    │   ├── MyFed.py
    │   └── __init__.py
    └── utils																	工具集
        ├── ConfigManager.py
        ├── Plot.py
        ├── ResultManager.py
        ├── Tools.py
        └── __init__.py

```

Time文件是一个多线程时间获取类的实现，Queue文件是因为mac的多线程queue部分功能未实现，对queue相关功能的实现。

## 配置文件

```text
{
  "global": {
    "experiment": "TMP/test/1",								实验路径/结果存放路径
    "data_file": "MNIST",                     数据集类文件
    "data_name": "MNIST",                     数据集类
    "iid": false,                             是否iid
    "client_num": 50													客户端数量
  },
  "server": {
    "epochs": 600,														服务器全局迭代次数
    "model_file": "CNN",											全局模型文件
    "model_name": "CNN",											全局模型类
    "checkin": {
      "checkin_interval": 600,
      "checkin_num": 200
    },
    "scheduler": {
      "scheduler_interval": 5,								调度间隔
      "schedule_file": "RandomSchedule",			调度算法文件
      "schedule_name": "RandomSchedule",      调度算法类
      "params": {															调度算法相关参数
        "c_ratio": 0.1,
        "schedule_interval": 5
      }
    },
    "updater": {
      "update_file": "MyFed",									聚合算法文件
      "update_name": "MyFed",									聚合算法类
      "params": {															聚合算法参数
        "a": 10,
        "b": 4,
        "alpha": 0.1,
        "r" : 1,
        "c" : 2,
        "d" : 2
      }
    }
  },
  "client": {
    "epochs": 2,															客户端迭代次数
    "batch_size": 50,
    "model_type": "CNN",
    "stale_file": "stale.txt"									延迟设置
  }
}
```
## 运行

直接运行`python main.py`即可，程序会自动读取根目录下的config.json文件，执行完后将结果储存到results下的指定路径下，并将配置文件一并存储。

也可以自行指定配置文件`python main.py config.json`，需要注意的是config.json的路径是基于根目录的，而非main.py。

## 特性

- [x] 异步联邦学习
- [x] 支持替换模型和数据集
- [x] 支持替换调度算法
- [x] 支持替换聚合算法
- [ ] 同步联邦学习
- [ ] 支持多GPU

## 代码尚存问题

目前框架里面有一个核心问题，客户端和服务器之间的通信使用的是`multiprocessing`的queue实现的，但是该队列在接收cuda张量后，当其他进程获取该张量，会导致内存溢出，程序异常退出。

这个bug是pytorch和queue导致的bug，暂时采取的解决方法是上传非cuda张量，聚合时再将其转为cuda张量，因此在添加聚合算法时，大致会需要出现如下代码：

```python
    updated_parameters = {}
    for key, var in client_weights.items():
        updated_parameters[key] = var.clone()
        if torch.cuda.is_available():
            updated_parameters[key] = updated_parameters[key].cuda()
```
