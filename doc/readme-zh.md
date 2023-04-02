# __Async-FL__

<img src="./pic/header.png" style="width:800px"></img>

![GitHub code size](https://img.shields.io/github/languages/code-size/desperadoccy/async-FL?style=flat-square&logo=github)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat-square&logo=github)](../license)
![python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![torch](https://img.shields.io/badge/torch-1.11.0-green?style=flat-square&logo=pytorch)

> This document is also available in: [中文](readme-zh.md) | [English](../readme.md)

>keywords: `federated-learning`, `asynchronous`, `synchronous`, `semi-asynchronous`, `personalized`

<details>
  <summary><b>目录</b></summary>
  <p>

- [初衷](#初衷)
- [git分支说明](#git分支说明)
- [基本配置](#基本配置)
- [运行](#运行)
  - [实验](#实验)
  - [Docker](#docker)
- [特性](#特性)
- [项目目录](#项目目录)
- [框架结构](#框架结构)
- [类解释](#类解释)
  - [接收器类](#接收器类)
  - [检查器类](#检查器类)
- [配置文件](#配置文件)
  - [异步配置文件](#异步配置文件)
  - [同步配置文件](#同步配置文件)
  - [半异步配置文件](#半异步配置文件)
- [添加新的算法](#添加新的算法)
  - [loss函数添加](#loss函数添加)
- [staleness设置](#staleness设置)
- [数据分布设置](#数据分布设置)
  - [iid](#iid)
  - [dirichlet non-iid](#dirichlet-non-iid)
  - [customize non-iid](#customize-non-iid)
    - [label distribution](#label-distribution)
    - [data distribution](#data-distribution)
- [客户端替换](#客户端替换)
- [多GPU](#多GPU)
- [代码尚存问题](#代码尚存问题)
- [Contributors](#contributors)
- [联系我](#联系我)
  </p>
</details>

## 初衷

本项目的初衷是我本科毕设期间需要完成搭建一个异步联邦学习框架，并且在其之上完成一些实验。

可当我去github尝试搜索项目时，发现异步联邦学习闭源之深，几乎没有开源项目。并且主流框架也基本不兼容异步，只支持同步FL。因此促生了该项目。

## git分支说明

master分支为主分支，代码为最新，但有部分commit是脏commit，不保证每个commit都能正常运行，建议使用打tag（版本号）的version

checkout分支保留了客户端会随着训练过程进行不断加入框架中，主分支已经移除该功能，checkout分支并不维护，只支持同步和异步。

## 基本配置

python3.8 + pytorch + macos

在linux进行过验证

支持单GPU，尚未进行多GPU优化

## 运行

### 实验
直接运行`python main.py`(fl下的main文件)即可，程序会自动读取根目录下的config.json文件，执行完后将结果储存到results下的指定路径下，并将配置文件一并存储。

也可以自行指定配置文件`python main.py config.json`，需要注意的是config.json的路径是基于根目录的，而非main.py。

根目录下的`config`文件夹提供了部分论文提出的算法文件配置，现提供如下算法实现：

```text
FedAvg
FedAsync
FedProx
FedAT
FedLC
FedDL
M-Step AsyncFL
```

### Docker

现在可以直接pull docker镜像进行运行，命令如下：

```shell
docker pull desperadoccy/async-fl
docker run -it async-fl config/FedAvg-config.json
```

类似地，支持传参config文件路径。
也可以自行build

```shell
cd docker
docker build -t async-fl .
docker run -it async-fl config/FedAvg-config.json 
```

## 特性

- [x] 异步联邦学习
- [x] 支持替换模型和数据集
- [x] 支持替换调度算法
- [x] 支持替换聚合算法
- [x] 支持替换loss函数
- [x] 支持替换客户端
- [x] 同步联邦学习
- [x] 半异步联邦学习
- [x] 提供test loss信息
- [x] 自定义标签异构
- [ ] 自定义数据异构
- [x] 支持dirichlet distribution
- [ ] 支持`Synthetic Non-Identical Client Data`生成;[相关论文](https://arxiv.org/pdf/1909.06335.pdf)
- [x] wandb可视化
- [ ] leaf相关数据集支持
- [x] 支持多GPU
- [x] docker部署

## 项目目录

<details>
  <summary><b>Project Directory</b></summary>
  <p>

```text
.
├── config                                    常见算法配置
│   ├── FedAT-config.json
│   ├── FedAsync-config.json
│   ├── FedAvg-config.json
│   └── FedProx-config.json
├── config.json                               配置文件
├── config_semi.json                          配置文件
├── config_semi_test.json                     配置文件
├── config_sync.json                          配置文件
├── config_sync_test.json                     配置文件
├── config_test.json                          配置文件
├── doc
│   ├── pic
│   │   ├── fedsemi.png
│   │   ├── framework.png
│   │   └── header.png
│   └── readme-zh.md
├── docker
│   └── Dockerfile
├── license
├── fedsemi.png
├── framework.png
├── readme.md
├── requirements.txt
└── src 
    ├── client                                客户端实现
    │   ├── AsyncClient.py                    异步客户端类
    │   ├── Client.py                         客户端基类
    │   ├── ProxClient.py
    │   ├── SemiClient.py
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
    │   ├── SchedulerThread.py                调度进程
    │   ├── UpdaterThread.py                  聚合进程
    │   └── __init__.py
    ├── fedsemi                               半异步联邦学习
    │   ├── QueueManager.py                   队列管理类
    │   ├── SchedulerThread.py                调度进程
    │   ├── SemiAsyncClientManager.py         客户端管理类
    │   ├── SemiAsyncServer.py                服务器类
    │   ├── UpdaterThread.py                  聚合进程
    │   ├── __init__.py
    │   ├── checker                           半异步检查器
    │   │   └── SemiAvgChecker.py
    │   ├── grouping                          分组（层）器
    │   │   ├── Grouping.py
    │   │   ├── NormalGrouping.py
    │   │   └── SimpleGrouping.py
    │   └── receiver                          半异步接收器
    │       └── SemiAvgReceiver.py
    ├── fedsync                               同步联邦学习
    │   ├── QueueManager.py                   消息队列管理类
    │   ├── SchedulerThread.py                调度进程
    │   ├── SyncClientManager.py              客户端管理类
    │   ├── SyncServer.py                     同步服务器类
    │   ├── UpdaterThread.py                  聚合进程
    │   ├── __init__.py
    │   ├── checker                           同步检查器
    │   │   └── AvgChecker.py
    │   └── receiver                          同步接收器
    │       └── AvgReceiver.py
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
    ├── results                               实验结果
    ├── schedule                              调度算法类
    │   ├── FullSchedule.py
    │   ├── RandomSchedule.py
    │   ├── RoundRobin.py
    │   └── __init__.py
    ├── test                                  测试用
    ├── update                                聚合算法类
    │   ├── AsyncAvg.py
    │   ├── FedAT.py
    │   ├── FedAsync.py
    │   ├── FedAvg.py
    │   ├── MyFed.py
    │   └── __init__.py
    └── utils                                 工具集
        ├── ConfigManager.py
        ├── IID.py
        ├── JsonTool.py
        ├── ModuleFindTool.py
        ├── ModelTraining.py
        ├── Plot.py
        ├── ProcessTool.py
        ├── Queue.py
        ├── Random.py
        ├── Time.py
        ├── Tools.py
        └── __init__.py
```

  </p>
</details>

utils包下的Time文件是一个多线程时间获取类的实现；Queue文件是因为mac的多线程queue部分功能未实现，对queue相关功能的实现。

## 框架结构

![error](pic/framework.png)

![error](pic/fedsemi.png)

## 类解释

### 接收器类

接收器是同步｜半异步联邦学习为了检查该轮全局迭代接收的更新是否满足设置的条件，如所有指定的客户端均已上传更新，满足条件则会触发updater进程进行全局聚合。

### 检查器类

同步｜半异步联邦学习中客户端完成训练后，会将权重上传给检查器类，检查起根据自身逻辑判断是否符合上传标准，选择接收或舍弃该更新。

## 配置文件

### 异步配置文件

<details>
  <summary><b>配置详情</b></summary>
  <p>

```text
{
  "wandb": {                                  wandb配置
    "enabled": true,                          是否启用
    "project": "non-iid test",                项目名称  
    "name": "1"                               本次运行结果
  },
  "global": {
    "multi_gpu": true,                        多gpu
    "mode": "async"                           同步｜异步｜半异步
    "experiment": "TMP/test/1",               实验路径/结果存放路径
    "stale": {                                延迟设置
      "step": 1,                              步长
      "shuffle": true,                        是否打乱
      "list": [10, 10, 10, 5, 5, 5, 5]        每个步长对应的客户端数
    },
    "dataset_path": "dataset.MNIST.MNIST",    Dataset路径
    "iid": {                                  non-iid设置
      "customize": true,                      启用自定义数据分布
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
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
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
      "updater_path": "update.FedAsync.FedAsync",
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
    "client_path": "client.AsyncClient.AsyncClient"     
  },
  "client": {
    "epochs": 2,                              客户端迭代次数
    "batch_size": 50,
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
    },
    "loss": "cross_entropy",                  loss函数
    "mu": 0.01,
    "optimizer": {                            优化器
      "path": "torch.optim.Adam",
      "weight_decay": 0.005,
      "lr": 0.01
    }
  }
}
```

  </p>
</details>

### 同步配置文件

<details>
  <summary><b>配置详情</b></summary>
  <p>

```text
{
  "wandb": {                                  wandb配置
    "enabled": true,                          是否启用
    "project": "non-iid test",                项目名称  
    "name": "1"                               本次运行结果
  },
  "global": {
    "multi_gpu": true,                        多gpu
    "mode": "sync"                            同步｜异步｜半异步
    "experiment": "TMP/test/1",               实验路径/结果存放路径
    "stale": {                                延迟设置
      "step": 1,                              步长
      "shuffle": true,                        是否打乱
      "list": [10, 10, 10, 5, 5, 5, 5]        每个步长对应的客户端数
    },
    "dataset_path": "dataset.MNIST.MNIST",    Dataset路径
    "iid": {                                  non-iid设置
      "customize": true,                      启用自定义数据分布
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
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
    },
    "scheduler": {
      "scheduler_interval": 5,                调度间隔
      "scheduler_path": "schedule.RandomSchedule.RandomSchedule",
      "params": {                             调度算法相关参数
        "c_ratio": 0.1,
        "schedule_interval": 5
      },
      "receiver": {
        "receiver_path": "fedsync.receiver.AvgReceiver.AvgReceiver", 
        "params": {
        }
      }
    },
    "updater": {
      "updater_path": "update.FedAvg.FedAvg",  
      "loss": "cross_entropy",                全局损失函数
      "params": {                             聚合算法参数
      }
    }
  },
  "client_manager": {
    "checker": {
      "checker_path": "fedsync.checker.AvgChecker.AvgChecker",        
      "params": {
      }
    },
    "client_path": "client.SyncClient.SyncClient"
  },
  "client": {
    "epochs": 2,                              客户端迭代次数
    "batch_size": 50,
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
    },
    "loss": "cross_entropy",                  loss函数
    "mu": 0.01,
    "optimizer": {                            优化器
      "path": "torch.optim.Adam",
      "weight_decay": 0,
      "lr": 0.01
    }
  }
}
```

  </p>
</details>

### 半异步配置文件

<details>
  <summary><b>配置详情</b></summary>
  <p>

```text
{
  "wandb": {                                  wandb配置
    "enabled": false,                         是否启用
    "project": "FedAT",                       项目名称  
    "name": "1"                               本次运行结果
  },
  "global": {
    "multi_gpu": true,                        多gpu
    "mode": "semi-async"                      同步｜异步｜半异步
    "experiment": "FedAT/1",                  实验路径/结果存放路径
    "stale": {                                延迟设置
      "step": 5,                              步长
      "shuffle": true,                        是否打乱
      "list": [10, 10, 10, 5, 5, 5, 5]        每个步长对应的客户端数
    },
    "dataset_path": "dataset.MNIST.MNIST",    Dataset路径
    "iid": {                                  non-iid设置
      "customize": true,                      启用自定义数据分布
      "label": {
        "step": 2,
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
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
    },
    "scheduler": {
      "scheduler_interval": 5,                调度间隔
      "scheduler_path": "schedule.RandomSchedule.RandomSchedule", 
      "params": {                             调度算法相关参数
        "c_ratio": 0.3,
        "schedule_interval": 0
      },
      "receiver": {
        "receiver_path": "fedsemi.receiver.SemiAvgReceiver.SemiAvgReceiver",
        "params": {
        }
      }
    },
    "updater": {
      "updater_path": "update.FedAT.FedAT",     
      "loss": "cross_entropy",                全局损失函数
      "params": {                             聚合算法参数
      },
      "group": {                              组内使用的聚合函数
        "updater_path": "update.FedAvg.FedAvg",
        "params": {
        }
      }
    },
    "grouping": {
      "grouping_path": "fedsemi.grouping.NormalGrouping.NormalGrouping",  
      "params": {                             分组管理参数
        "step": 5
      }
    }
  },
  "client_manager": {
    "checker": {
      "checker_path": "fedsemi.checker.SemiAvgChecker.SemiAvgChecker", 
      "params": {
      }
    },
    "client_path": "client.SemiClient.SemiClient"
  },
  "client": {
    "epochs": 2,                              客户端迭代次数
    "batch_size": 50,
    "model": {
      "path": "model.CNN.CNN",
      "params": {}
    }, 
    "loss": "cross_entropy",                  loss函数
    "mu": 0.01,
    "optimizer": {                            优化器
      "path": "torch.optim.SGD",
      "weight_decay": 0,
      "lr": 0.01
    }
  }
}
```

  </p>
</details>

## 添加新的算法

需要让客户端/服务器调用自己的算法或实现类，（注意：所有的算法实现必须以类的形式），需要以下几个步骤：

* 在对应的位置加入自己的实现（dataset、model、schedule、update、client、loss）
* 在对应包的`__init__.py`文件下导入该类，例如`from model import CNN`
* 在配置文件申明，`model.path`等对应的是新的算法所在路径。

另外，算法里需要使用到的参数均可在配置项`params`中申明。

现在`model`、`optim`、`loss`模块支持引入`torch`等模块自带实现，例：

```json
"model": {
      "path": "torchvision.models.resnet18",
      "params": {
        "pretrained": true,
        "num_classes": 10 
      }
}
```

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

## staleness设置

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

## 数据分布设置

### iid

当iid设置为true时（其实false也是默认为iid），会以iid的方式将数据分配给各客户端。

```json
"iid": true
```

### dirichlet non-iid

当`iid`中`customize`设置为false或者不设置时，会以dirichlet分布的方式将数据分配给各客户端。
其中beta是dirichlet分布的参数。

```json
"iid": {
    "customize": false,
    "beta": 0.5
}
```

或者

```json
"iid": {
    "beta": 0.5
}
```

### customize non-iid

customize non-iid设置分为两部分，一个是标签的non-iid设置，一个是数据量的non-iid设置。目前数据量仅提供随机生成，在未来的版本中将引入个性化设置。
在启用customize设置时，需要将`customize`设置为true并分别对`label`和`data`进行设置

```json
"iid": {
    "customize": true
}
```

#### label distribution

label的设置stale的设置类似，支持三种方式，其一为配置文件中提到的

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

其三为一维数组，该一维数组为每个客户端拥有的标签数，该数组长度应和客户端数量一致。

```json
"label": {
  "list": [4, 5, 10, 1, 2, 3, 4]
}
```

上述配置即客户端0拥有4个标签数据，客户端1拥有5个标签数据...以此类推。

目前label_iid生成的随机化分为两种方法，一种纯随机化，这种情况可能会导致所有客户端均缺少一个标签，导致精度下降（虽然概率极低），另一种方式采用洗牌算法，保证每个标签均会选到，这也会导致无法生成标签分布不均匀的数据情况。洗牌算法的开关由`shuffle`控制，示例如下：

```json
"label": {
  "shuffle": true,
  "list": [4, 5, 10, 1, 2, 3, 4]
}
```

#### data distribution

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

## 多GPU

本项目的多GPU特性并不是多GPU并行计算，各客户端训练依旧在单GPU上，但宏观上客户端运行在多个GPU上，也就是每个客户端的训练任务会平均分布到`程序可见`的GPU上，每个客户端绑定的GPU是在初始化时就指定好的，并不是每轮训练时指定，因此依旧会出现各GPU负载严重不均的可能情况。
该特性通过global下的`multi_gpu`控制开关。

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

## Contributors

<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/desperadoccy">
            <img src="https://avatars.githubusercontent.com/u/44546125?v=4" width="100;" alt="desperadoccy"/>
            <br />
            <sub><b>Desperadoccy</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/jzj007">
            <img src="https://avatars.githubusercontent.com/u/73173984?v=4" width="100;" alt="jzj007"/>
            <br />
            <sub><b>Jzj007</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

## 联系我

QQ: 527707607

邮箱: desperado@qq.com

欢迎对项目提出建议～
