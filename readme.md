# 联邦学习简易框架

>keywords: `federated-learning`, `asynchronous`, `synchronous`

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
├── config                                    常见算法配置
│   ├── FedAsync-config.json
│   ├── FedAvg-config.json
│   └── FedProx-config.json
├── config_sync.json                          配置文件   
├── config.json                               配置文件    
├── framework.png                            
├── readme.md                                 
├── requirements.txt
└── src 
    ├── checker                               上传器类
    │   ├── AvgChecker.py
    │   └── __init__.py
    ├── client                                客户端实现
    │   ├── AsyncClient.py                    异步客户端类
    │   ├── Client.py                         客户端基类
    │   ├── ProxClient.py
    │   ├── SyncClient.py                     同步客户端类
    │   └── __init__.py
    ├── data                                  数据集下载位置
    ├── dataset                               数据集类
    │   ├── CIFAR10.py
    │   ├── MNIST.py
    │   ├── FashionMNIST.py
    │   └── __init__.py
    ├── exception                             异常类
    │   ├── ClientSumError.py
    │   └── __init__.py
    ├── fedasync                              异步联邦学习
    │   ├── AsyncClientManager.py             客户端管理类
    │   ├── AsyncServer.py                    异步服务器类
    │   ├── CheckInThread.py                  CheckIn进程
    │   ├── SchedulerThread.py                调度进程
    │   ├── UpdaterThread.py                  聚合进程
    │   └── __init__.py
    ├── fedsync                               同步联邦学习
    │   ├── CheckInThread.py                  CheckIn进程
    │   ├── QueueManager.py                   消息队列管理类
    │   ├── SchedulerThread.py                调度进程
    │   ├── SyncClientManager.py              客户端管理类
    │   ├── SyncServer.py                     同步服务器类
    │   ├── UpdaterThread.py                  聚合进程
    │   └── __init__.py
    ├── fl                                    fl主函数
    │   ├── __init__.py
    │   ├── main.py
    │   └── wandb                             wandb运行文件夹
    ├── loss                                  loss函数实现
    │   └── __init__.py
    ├── model                                 模型类
    │   ├── CNN.py
    │   ├── ConvNet.py
    │   └── __init__.py
    ├── receiver                              接收器类
    │   ├── AvgReceiver.py
    │   └── __init__.py
    ├── results                               实验结果
    ├── schedule                              调度算法类
    │   ├── RandomSchedule.py
    │   └── __init__.py
    ├── test                                  测试用
    ├── update                                聚合算法类
    │   ├── Avg.py
    │   ├── FedAsync.py
    │   ├── MyFed.py
    │   └── __init__.py
    └── utils                                 工具集
        ├── ConfigManager.py
        ├── JsonTool.py
        ├── ModuleFindTool.py
        ├── Plot.py
        ├── Queue.py
        ├── Time.py
        ├── Tools.py
        └── __init__.py

```

Time文件是一个多线程时间获取类的实现，Queue文件是因为mac的多线程queue部分功能未实现，对queue相关功能的实现。

## 框架结构

![error](framework.png)

## 类解释

### 接收器类

接收器是同步联邦学习为了检查该轮全局迭代接收的更新是否满足设置的条件，如所有指定的客户端均已上传更新，满足条件则会触发updater进程进行全局聚合。

### 上传器类

同步联邦学习中客户端完成训练后，会将权重上传给上传器类，上传器根据自身逻辑判断是否符合上传标准，选择接收或舍弃该更新。

## 配置文件

### 异步配置文件

```text
{
  "wandb": {                                  wandb配置
    "enabled": true,                          是否启用
    "project": "non-iid test",                项目名称  
    "name": "1"                               本次运行结果
  },
  "global": {
    "mode": "async"                           同步｜异步
    "experiment": "TMP/test/1",               实验路径/结果存放路径
    "stale": {                                延迟设置
      "step": 1,                              步长
      "shuffle": true,                        是否打乱
      "list": [10, 10, 10, 5, 5, 5, 5]        每个步长对应的客户端数
    },
    "data_file": "MNIST",                     数据集类文件
    "data_name": "MNIST",                     数据集类
    "iid": {                                  non-iid设置
      "label": {
        "step": 1,
        "list": [10, 10, 30]
      },
      "data": {
        "max": 200,
        "min": 200
      }
    },
    "client_num": 50                          客户端数量
  },
  "server": {
    "epochs": 600,                            服务器全局迭代次数
    "model_file": "CNN",                      全局模型文件
    "model_name": "CNN",                      全局模型类
    "checkin": {
      "checkin_interval": 600,
      "checkin_num": 200
    },
    "scheduler": {
      "scheduler_interval": 5,                调度间隔
      "schedule_file": "RandomSchedule",      调度算法文件
      "schedule_name": "RandomSchedule",      调度算法类
      "params": {                             调度算法相关参数
        "c_ratio": 0.1,
        "schedule_interval": 5
      }
    },
    "updater": {
      "update_file": "MyFed",                 聚合算法文件
      "update_name": "MyFed",                 聚合算法类
      "loss": "cross_entropy",                全局损失函数
      "params": {                             聚合算法参数
        "a": 10,
        "b": 4,
        "alpha": 0.1,
        "r" : 1,
        "c" : 2,
        "d" : 2
      }
    }
  },
  "client_manager": {
    "client_file": "AsyncClient",             客户端文件
    "client_name": "AsyncClient"              客户端类
  },
  "client": {
    "epochs": 2,                              客户端迭代次数
    "batch_size": 50,
    "model_file": "CNN",                      本地模型文件
    "model_name": "CNN",                      本地模型类
    "loss": "cross_entropy",                  loss函数
    "optimizer": {                            优化器
      "name": "Adam",
      "weight_decay": 0.005,
      "lr": 0.01
    }
  }
}
```

### 同步配置文件

```text
{
  "wandb": {                                  wandb配置
    "enabled": true,                          是否启用
    "project": "non-iid test",                项目名称  
    "name": "1"                               本次运行结果
  },
  "global": {
    "mode": "async"                           同步｜异步
    "experiment": "TMP/test/1",               实验路径/结果存放路径
    "stale": {                                延迟设置
      "step": 1,                              步长
      "shuffle": true,                        是否打乱
      "list": [10, 10, 10, 5, 5, 5, 5]        每个步长对应的客户端数
    },
    "data_file": "MNIST",                     数据集类文件
    "data_name": "MNIST",                     数据集类
    "iid": {                                  non-iid设置
      "label": {
        "step": 1,
        "list": [10, 10, 30]
      },
      "data": {
        "max": 200,
        "min": 200
      }
    },
    "client_num": 50                          客户端数量
  },
  "server": {
    "epochs": 600,                            服务器全局迭代次数
    "model_file": "CNN",                      全局模型文件
    "model_name": "CNN",                      全局模型类
    "checkin": {
      "checkin_interval": 600,
      "checkin_num": 200
    },
    "scheduler": {
      "scheduler_interval": 5,                调度间隔
      "schedule_file": "RandomSchedule",      调度算法文件
      "schedule_name": "RandomSchedule",      调度算法类
      "params": {                             调度算法相关参数
        "c_ratio": 0.1,
        "schedule_interval": 5
      },
      "receiver": {
        "receiver_file": "AvgReceiver",       接收器文件
        "receiver_name": "AvgReceiver"        接收器类 
        "params": {
        }
      }
    },
    "updater": {
      "update_file": "MyFed",                 聚合算法文件
      "update_name": "MyFed",                 聚合算法类
      "loss": "cross_entropy",                全局损失函数
      "params": {                             聚合算法参数
        "a": 10,
        "b": 4,
        "alpha": 0.1,
        "r" : 1,
        "c" : 2,
        "d" : 2
      }
    }
  },
  "client_manager": {
    "checker": {
      "checker_file": "AvgChecker",           检查器文件 
      "checker_name": "AvgChecker",           检查器类
      "params": {
      }
    },
    "client_file": "SyncClient",
    "client_name": "SyncClient"
  },
  "client": {
    "epochs": 2,                              客户端迭代次数
    "batch_size": 50,
    "model_file": "CNN",                      本地模型文件
    "model_name": "CNN",                      本地模型类
    "loss": "cross_entropy",                  loss函数
    "optimizer": {                            优化器
      "name": "Adam",
      "weight_decay": 0,
      "lr": 0.01
    }
  }
}
```

## 运行

### 实验
直接运行`python main.py`(fl下的main文件)即可，程序会自动读取根目录下的config.json文件，执行完后将结果储存到results下的指定路径下，并将配置文件一并存储。

也可以自行指定配置文件`python main.py config.json`，需要注意的是config.json的路径是基于根目录的，而非main.py。

根目录下的`config`文件夹提供了部分论文提出的算法文件配置，现提供如下算法实现：

```text
FedAvg
FedAsync
FedProx
```

## 特性

- [x] 异步联邦学习
- [x] 支持替换模型和数据集
- [x] 支持替换调度算法
- [x] 支持替换聚合算法
- [x] 支持替换loss函数
- [x] 支持替换客户端
- [x] 同步联邦学习
- [ ] 半异步联邦学习
- [x] 提供test loss信息
- [ ] 接收客户端loss信息
- [x] 自定义标签异构
- [ ] 自定义数据异构
- [ ] 支持`Synthetic Non-Identical Client Data`生成;[相关论文](https://arxiv.org/pdf/1909.06335.pdf)
- [x] wandb可视化
- [ ] leaf相关数据集支持
- [ ] 支持多GPU

## 添加新的算法

需要让客户端/服务器调用自己的算法或实现类，（注意：所有的算法实现必须以类的形式），需要以下几个步骤：

* 在对应的位置加入自己的实现（dataset、model、schedule、update、client、loss）
* 在对应包的`__init__.py`文件下导入该类，例如`from model import CNN`
* 在配置文件申明，`model_file`等对应的是新的算法所在文件名，`model_name`等对应的是新的算法的类。

另外，算法里需要使用到的参数均可在配置项`params`中申明。

### loss函数添加

loss函数可以选择torch自带算法，也可以自行实现，自行实现和上述步骤基本相同，在配置项中需进行如下修改：

```json
"client": {
    "loss": {
        "loss_file": "my_loss",
        "loss_name": "my_loss"
  }
}
```

## stale设置

stale支持三种设置，其一是上述配置文件中提到的

```json
"stale": {
      "step": 5,
      "shuffle": true,
      "list": [10, 10, 10, 5, 5, 5, 5]
    }
```

程序会根据提供的`step`和`list`生成一串随机整数，例如上述代码，程序会生成10个0，10个(0，5)，10个[5,10)......，并会根据`shuffle`判断是否进行打乱。最后将随机数串赋给各客户端，客户端根据数值在每轮训练结束后，自动sleep对应秒。在存储json文件至实验结果时，该设置会自动转为其三。

其二是设置为false，程序会给各客户端延迟设置为0。

```json
"stale": false
```

其三是随机数列表，程序直接会将列表指定延迟设置给客户端。

```json
"stale": [1, 2, 3, 1, 4]
```

## non-iid设置

non-iid设置分为两部分，一个是标签的non-iid设置，一个是数据量的non-iid设置。目前数据量仅提供随机生成，在未来的版本中将引入个性化设置。

当iid设置为true时（其实false也是默认为iid），会以iid的方式将数据分配给各客户端。

```json
"iid": true
```

label的设置stale的设置类似，支持两种方式，其一为配置文件中提到的

```json
"label": {
    "step": 1,
    "list": [10, 10, 30]
}
```

其上配置程序会生成10个拥有1个标签数据的客户端，10个拥有2个标签数据的客户端，30个拥有3个标签数据的客户端
step是标签数量的步长，当step为2时，程序会生成10个拥有1个标签数据的客户端，10个拥有3个标签数据的客户端，30个拥有5个标签数据的客户端

其二为随机数二维数组，程序将二维数组直接设置给客户端

```json
"label": {
    "0": [1, 2, 3, 8],
    "1": [2, 4],
    "2": [4, 7],
    "3": [0, 2, 3, 6, 9],
    "4": [5]
}
```

data的设置比较简单，目前有两种方式，其一为空

```json
"data": {}
```

也就是不对数据量进行非独立同分布设置。

其二为配置文件中提到的

```json
"data": {
    "max": 500,
    "min": 400
}
```

也就是说客户端的数据量范围在400-500，程序会自动平均分配到各标签

数据量分布还较初始，之后将会逐步完善

## 客户端替换

目前客户端替换需要继承`AsyncClient`或`SyncClient`，新增的参数通过client配置项传入类中。

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

## 联系我

QQ: 527707607

邮箱: desperado@qq.com

欢迎对项目提出建议～
