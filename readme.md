<img src="./doc/pic/header.png" style="width:800px"></img>

![GitHub code size](https://img.shields.io/github/languages/code-size/desperadoccy/async-FL?style=flat-square&logo=github)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat-square&logo=github)](./license)
![python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![torch](https://img.shields.io/badge/torch-1.11.0-green?style=flat-square&logo=pytorch)

> This document is also available in: [中文](doc/readme-zh.md) | [English](readme.md)

> keywords: `federated-learning`, `asynchronous`, `synchronous`, `semi-asynchronous`, `personalized`

> 
<details>
  <summary><b>Table of Contents</b></summary>
  <p>

- [Brief](#brief)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [Environment](#environment)
  - [Experiments](#experiments)
  - [Docker](#docker)
- [Features](#features)
- [Add new Methods](#add-new-methods)
- [Existing Bugs](#existing-bugs)
- [Contributors](#contributors)
- [Citation](#citation)
- [Contact Us](#contact-us)

  </p>
</details>

## Brief

One code adapts to multiple operating modes: [`thread`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#thread), [`process`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#process),  [`MPMT`](https://github.com/NUAA-SmartSensing/FedModule/wiki/mode#mpmt), [`distributed`](https://github.com/NUAA-SmartSensing/async-FL/wiki/mode#distributed).

One-click start; change the experimental environment without modifying the code.

Support random seeds for reproducible experiments.

Redesigned the FL framework to be module with high extensibility, supporting various mainstream federated learning paradigms: `synchronous`, `asynchronous`, `semi-asynchronous`, `personalized`, etc.

With wandb, synchronize experimental data to the cloud, avoiding data loss.

For more project information, please see the [wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki).

## Requirements

python3.8 + pytorch + linux

It has been validated on macOS.

It supports single GPU and Multi-GPU.

## Getting Started

### Environment

Install dependencies on an existing python environment using `pip install -r requirements.txt`

or

Create a new python environment using conda:

```shell
conda env create -f environment.yml
```

### Experiments
You can run `python main.py` (the main file in the fl directory) directly. The program will automatically read the `config.json` file in the root directory and store the results in the specified path under `results`, along with the configuration file.

You can also specify the configuration file by `python main.py ../../config.json`. Please note that the path of `config.json` is relative to the `main.py`.

The `config` folder in the root directory provides some algorithm configuration files proposed in papers. The following algorithm implementations are currently available:

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

more methods to refer to the [wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki/%E7%8E%B0%E6%9C%89%E7%AE%97%E6%B3%95)

### Docker

Now you can directly pull and run a Docker image, the command is as follows:

```shell
docker pull desperadoccy/async-fl
docker run -it async-fl config/FedAvg-config.json
```

Similarly, it supports passing a config file path as a parameter. You can also build the Docker image yourself.

```shell
cd docker
docker build -t async-fl .
docker run -it async-fl config/FedAvg-config.json 
```

## Features

- [x] Asynchronous Federated Learning
- [x] Support model and dataset replacement
- [x] Support scheduling algorithm replacement
- [x] Support aggregation algorithm replacement
- [x] Support loss function replacement
- [x] Support client replacement
- [x] Synchronous federated learning
- [x] Semi-asynchronous federated learning
- [x] Provide test loss information
- [x] Custom label heterogeneity
- [x] Custom data heterogeneity
- [x] Support Dirichlet distribution
- [x] wandb visualization
- [x] Support for multiple GPUs
- [x] Docker deployment
- [x] Process thread switching

## Add new methods

Please refer to the [wiki](https://github.com/NUAA-SmartSensing/async-FL/wiki/module_guide#开发模块)

## Existing Bugs

Currently, there is a core issue in the framework that the communication between clients and servers is implemented using the `multiprocessing` queues. However, when a CUDA tensor is received by the queue and retrieved by other threads, it can cause a memory leak and may cause the program to crash.

This bug is caused by PyTorch and the multiprocessing queue, and the current solution is to upload non-CUDA tensors to the queue and convert them to CUDA tensors during aggregation. Therefore, when adding aggregation algorithms, the following code will be needed:

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

## Citation

Please cite our paper in your publications if this code helps your research.

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

## Contact us

We created a QQ group to discuss the asyncFL framework and FL, welcome everyone to join~~

Here is the group number:

895896624

![group_number](./doc/pic/group.png)

QQ: 527707607

email: desperado@qq.com

Welcome to provide suggestions for the project~

if you'd like contribute to this project, please contact us.
