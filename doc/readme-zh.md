
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

- [简介](#简介)
- [基本配置](#基本配置)
- [运行](#运行)
  - [环境](#环境)
  - [实验](#实验)
  - [Docker](#docker)
- [特性](#特性)
- [添加新的算法](#添加新的算法)
- [代码尚存问题](#代码尚存问题)
- [Contributors](#contributors)
- [引用][#引用]
- [联系我](#联系我)
  </p>
</details>

## 简介

一份代码适配多种运行模式：[`thread`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#thread), [`process`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#process),  [`MPMT`](https://github.com/NUAA-SmartSensing/FedModule/wiki/mode#mpmt), [`distributed`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#distributed)。

一键启动，更改实验环境无需修改代码。

支持随机种子，可重复实验。

重新将FL模块化设计，具有高扩展性，支持各类主流联邦学习模式：`同步`、`异步`、`半异步`、`个性化`等等。

wandb将实验数据同步云端，不必担心数据丢失。

更多项目信息请查看[wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki)

## 基本配置

python3.8 + pytorch + linux

在macOS进行过验证

支持多GPU优化

## 运行

### 环境

通过`pip install -r requirements.txt`在已有的python环境中安装依赖

或者

通过conda创建新的python环境

```shell
conda env create -f environment.yml
```

### 实验
直接运行`python main.py`(fl下的main文件)即可，程序会自动读取根目录下的config.json文件，执行完后将结果储存到results下的指定路径下，并将配置文件一并存储。

也可以自行指定配置文件`python main.py ../../config.json`，需要注意的是config.json的路径是基于`main.py`的。

根目录下的`config`文件夹提供了部分论文提出的算法文件配置，现提供如下算法实现：

```text
Centralized Learning
FedAvg
FedAsync
FedProx
FedAT
FedLC
FedDL
M-Step AsyncFL
FedBuff
FedAdam
FedNova
FedBN
TWAFL
```

更多算法请查看[wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki/%E7%8E%B0%E6%9C%89%E7%AE%97%E6%B3%95)


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
- [x] 同步联邦学习
- [x] 半异步联邦学习
- [x] 支持替换模型和数据集
- [x] 支持替换调度算法
- [x] 支持替换聚合算法
- [x] 支持替换loss函数
- [x] 支持替换客户端
- [x] 自定义标签异构
- [x] 自定义数据异构
- [x] 支持dirichlet distribution
- [x] wandb可视化
- [x] 支持多GPU
- [x] docker部署
- [x] 进线程切换

## 添加新的算法

请查看[wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki/module_guide#开发模块)

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
    </td>
    <td align="center">
        <a href="https://github.com/cauchyguo">
            <img src="https://avatars.githubusercontent.com/u/41313807?v=4" width="100;" alt="cauchyguo"/>
            <br />
            <sub><b>Cauchy</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

## 引用

如果代码对你的研究有帮助的话，请引用我们的文章。

```latex
@misc{chen2024fedmodulemodularfederatedlearning,
      title={FedModule: A Modular Federated Learning Framework}, 
      author={Chuyi Chen and Zhe Zhang and Yanchao Zhao},
      year={2024},
      eprint={2409.04849},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.04849}, 
}
```

## 联系我

我们创建了一个QQ群用来讨论asyncFL框架和FL，欢迎大家加入~~

以下是群号:

895896624

![group_number](./pic/group.png)

QQ: 527707607

邮箱: desperado@qq.com

欢迎对项目提出建议～
