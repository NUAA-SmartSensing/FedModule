# Async-FL

<img src="./doc/pic/header.png" style="width:800px"></img>

![GitHub code size](https://img.shields.io/github/languages/code-size/desperadoccy/async-FL?style=flat-square&logo=github)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat-square&logo=github)](./license)
![python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![torch](https://img.shields.io/badge/torch-1.11.0-green?style=flat-square&logo=pytorch)

> This document is also available in: [中文](doc/readme-zh.md) | [English](readme.md)

> keywords: `federated-learning`, `asynchronous`, `synchronous`, `semi-asynchronous`, `personalized`

<details>
  <summary><b>Table of Contents</b></summary>
  <p>

- [Original Intention](#original-intention)
- [Git Branch Description](#git-branch-description)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [Experiments](#experiments)
  - [Docker](#docker)
- [Features](#features)
- [Project Directory](#project-directory)
- [Framework](#Framework)
- [Code Explanations](#code-explanations)
  - [Receiver Class](#receiver-class)
  - [Checker Class](#checker-class)
- [Configuration](#Configuration)
  - [Asynchronous Configuration](#asynchronous-configuration)
  - [Synchronous Configuration](#synchronous-configuration)
  - [Semi-aynchronous Configuration](#semi-aynchronous-configuration)
  - [Parameter explanation](#parameter-explanation)
- [Adding New Algorithm](#adding-new-algorithm)
  - [Adding Loss Function](#adding-loss-function)
- [Staleness Settings](#staleness-settings)
- [Data Distribution Settings](#data-distribution-settings)
  - [iid](#iid)
  - [dirichlet non-iid](#dirichlet-non-iid)
  - [customize non-iid](#customize-non-iid)
    - [label distribution](#label-distribution)
    - [data distribution](#data-distribution)
- [Adding New Client Class](#adding-new-client-class)
- [Multi-GPU](#multi-gpu)
- [Existing Bugs](#existing-bugs)
- [Contributors](#contributors)
- [Contact Us](#contact-us)

  </p>
</details>

## Original Intention

The initial intention of this project is to build an asynchronous federated learning framework and conduct experiments on it during my undergraduate thesis.

However, when I tried to search for related open-source projects on GitHub, I found that the field of asynchronous federated learning is quite closed-source, with almost no open-source projects available. Additionally, mainstream frameworks also lack compatibility with asynchronous FL and only support synchronous FL. Thus, this project was born.

## Git Branch Description

The master branch is the main branch with the latest code, but some of the commits are dirty commits and not guaranteed to run properly. It is recommended to use tagged versions for better stability.

The checkout branch retains the functionality of adding clients to the system during the training process, which has been removed in the main branch. The checkout branch is not actively maintained and only supports synchronous and asynchronous FL.

## Requirements

python3.8 + pytorch + macos

It has been validated on Linux.

It supports single GPU and Multi-GPU.

## Getting Started

### Experiments
You can run `python main.py` (the main file in the fl directory) directly. The program will automatically read the `config.json` file in the root directory and store the results in the specified path under `results`, along with the configuration file.

You can also specify the configuration file by `python main.py config.json`. Please note that the path of `config.json` is relative to the root directory, not `main.py`.

The `config` folder in the root directory provides some algorithm configuration files proposed in papers. The following algorithm implementations are currently available:

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
- [ ] Custom data heterogeneity
- [x] Support Dirichlet distribution
- [x] wandb visualization
- [ ] Support for leaf-related datasets
- [x] Support for multiple GPUs
- [x] Docker deployment

## Project Directory

<details>
  <summary><b>Project Directory</b></summary>
  <p>

```text
.
├── config                                    Common algorithm configuration files
│   ├── FedAT-config.json
│   ├── FedAsync-config.json
│   ├── FedAvg-config.json
│   ├── FedDL-config.json
│   ├── FedLC-config.json
│   ├── FedProx-config.json
│   ├── MSTEPAsync-config.json
│   ├── config.json
│   └── model_config
│       ├── CIFAR10-config.json
│       ├── ResNet18-config.json
│       └── ResNet50-config.json
├── config.json
├── config_semi.json
├── config_semi_test.json
├── config_sync.json
├── config_sync_test.json
├── config_test.json
├── doc
│   ├── params.docx
│   ├── pic
│   │   ├── fedsemi.png
│   │   ├── framework.png
│   │   └── header.png
│   ├── readme-zh.md
│   └── 参数.docx
├── docker
│   └── Dockerfile
├── license
├── readme.md
├── requirements.txt
└── src
    ├── checker                                checker implementation
    │   ├── AllChecker.py
    │   ├── CheckerCaller.py
    │   ├── SyncChecker.py
    │   └── __init__.py
    ├── client                                 client implementation
    │   ├── ActiveClient.py
    │   ├── Client.py
    │   ├── DLClient.py
    │   ├── NormalClient.py
    │   ├── ProxClient.py
    │   ├── SemiClient.py
    │   ├── TestClient.py
    │   └── __init__.py
    ├── clientmanager                           client manager implementation
    │   ├── BaseClientManager.py
    │   ├── NormalClientManager.py
    │   └── __init__.py
    ├── compressor                              compressor algorithm class
    │   ├── QSGD.py
    │   └── __init__.py
    ├── data
    ├── dataset
    │   ├── CIFAR10.py
    │   ├── FashionMNIST.py
    │   ├── MNIST.py
    │   └── __init__.py
    ├── exception
    │   ├── ClientSumError.py
    │   └── __init__.py
    ├── fl                                       wandb running directory
    │   ├── __init__.py
    │   ├── main.py
    │   └── wandb
    ├── group                                    group algorithm class
    │   ├── AbstractGroup.py
    │   ├── DelayGroup.py
    │   ├── GroupCaller.py
    │   ├── OneGroup.py
    │   └── __init__.py
    ├── groupmanager                             group manager implementation
    │   ├── BaseGroupManager.py
    │   ├── NormalGroupManager.py
    │   └── __init__.py
    ├── loss                                     loss algorithm class
    │   ├── FedLC.py
    │   ├── LossFactory.py
    │   └── __init__.py
    ├── model
    │   ├── CNN.py
    │   └── __init__.py
    ├── numgenerator                             num generator algorithm class
    │   ├── AbstractNumGenerator.py
    │   ├── NumGeneratorFactory.py
    │   ├── StaticNumGenerator.py
    │   └── __init__.py
    ├── queuemanager                             queuemanager implementation
    │   ├── AbstractQueueManager.py
    │   ├── BaseQueueManger.py
    │   ├── QueueListManager.py
    │   ├── SingleQueueManager.py
    │   └── __init__.py
    ├── receiver                                 receiver implementation
    │   ├── MultiQueueReceiver.py
    │   ├── NoneReceiver.py
    │   ├── NormalReceiver.py
    │   ├── ReceiverCaller.py
    │   └── __init__.py
    ├── results
    ├── schedule                                 scheduling algorithm class
    │   ├── AbstractSchedule.py
    │   ├── FullSchedule.py
    │   ├── NoSchedule.py
    │   ├── RandomSchedule.py
    │   ├── RoundRobin.py
    │   ├── ScheduleCaller.py
    │   └── __init__.py
    ├── scheduler                                scheduler implementation
    │   ├── AsyncScheduler.py
    │   ├── BaseScheduler.py
    │   ├── SemiAsyncScheduler.py
    │   ├── SyncScheduler.py
    │   └── __init__.py
    ├── server                                   server implementation
    │   ├── AsyncServer.py
    │   ├── BaseServer.py
    │   ├── SemiAsyncServer.py
    │   ├── SyncServer.py
    │   └── __init__.py
    ├── test                                     for test
    │   ├── __init__.py
    │   ├── test.ipynb
    │   └── test.py
    ├── update                                   update algorithm class
    │   ├── AbstractUpdate.py
    │   ├── AsyncAvg.py
    │   ├── FedAT.py
    │   ├── FedAsync.py
    │   ├── FedAvg.py
    │   ├── FedDL.py
    │   ├── StepAsyncAvg.py
    │   ├── UpdateCaller.py
    │   └── __init__.py
    ├── updater                                 updater implementation
    │   ├── AsyncUpdater.py
    │   ├── BaseUpdater.py
    │   ├── SemiAsyncUpdater.py
    │   ├── SyncUpdater.py
    │   └── __init__.py
    └── utils
        ├── ConfigManager.py
        ├── GlobalVarGetter.py
        ├── IID.py
        ├── JsonTool.py
        ├── ModelTraining.py
        ├── ModuleFindTool.py
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

The "Time" file under the "utils" package is an implementation of a multi-threaded time acquisition class, and the "Queue" file is an implementation of related functionalities for the "queue" module, as some functionalities of the "queue" module are not yet implemented on macOS.
## Framework

![error](doc/pic/framework.png)

![error](doc/pic/fedsemi.png)

## Code Explanations

### Receiver Class

The receiver in synchronous and semi-asynchronous federated learning is used to check whether the updates received during the current global iteration meet the conditions set, such as whether all designated clients have uploaded their updates. If the conditions are met, the updater process will be triggered to perform global aggregation.

### Checker Class

In synchronous and semi-asynchronous federated learning, after a client completes its training, it will upload its weights to the uploader class, which will determine whether the update meets the upload criteria based on its own logic, and decide whether to accept or discard the update.

## Configuration

### Configuration

[async mdoe example](config/FedAsync-config.json)

[sync mdoe example](config/FedAvg-config.json)

[semi-async mdoe example](config/FedAT-config.json)

### Parameter explanation

<details>
  <summary><b>Parameter explanation</b></summary>
  <p>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 align=left
 style='border-collapse:collapse;border:none;margin-left:6.75pt;margin-right:
 6.75pt'>
 <tr style='height:1.0cm'>
  <td colspan=5 style='border:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>parameters</span></p>
  </td>
  <td width=47 style='width:34.9pt;border:solid windowtext 1.0pt;border-left:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>type</span></p>
  </td>
  <td width=496 style='width:371.85pt;border:solid windowtext 1.0pt;border-left:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>explanation</span><span
  lang=EN-US style='font-family:Kai'>s</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=3 style='border:solid windowtext 1.0pt;border-top:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>wandb</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>enabled</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>bool</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>whether to enable wandb</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>project</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>project name</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>name</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the name of this run</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=8 style='border:solid windowtext 1.0pt;border-top:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>global</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>use_file_system</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>bool</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>whether to enable the file system as the
  torch multi-thread sharing strategy</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>multi_gpu</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>bool</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>whether to enable multi-GPU, detailed </span><a
  href="#multi-gpu"><span lang=EN-US style='font-family:Kai'>explanation</span></a></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>experiment</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the name of this run</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>stale</span></p>
  </td>
  <td colspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><a href="#staleness_settings"><span lang=EN-US
  style='font-family:Kai'>explanation</span></a></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dataset</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the dataset</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>iid</span></p>
  </td>
  <td colspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'><a href="#data-distribution-settings">explanation</a></span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>client_num</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>int</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>client num</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=15 style='border:solid windowtext 1.0pt;border-top:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>server</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string </span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of server</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>epochs</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>int</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>global epoch</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>model</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>the path of the model</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td rowspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>scheduler</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>the path of the</span><span
  lang=EN-US style='font-family:Kai'> scheduler</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=2 rowspan=2 style='border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:
  Kai'>schedule</span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the schedule</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>other_params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>*</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>other parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td rowspan=7 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>updater</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the updater</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=2 rowspan=2 style='border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>update</span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the update</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>loss</span></p>
  </td>
  <td colspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><a
  href="#adding-loss-function"><span lang=EN-US style='font-family:Kai'>explanation</span></a></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>num_generator</span></p>
  </td>
  <td width=542 colspan=2 style='width:406.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'><a href="#num_generator">explanation</a></span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=2 rowspan=2 style='border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>group</span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the updater</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td style='border:solid windowtext 1.0pt;border-top:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>client_manager</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the client</span><span
  lang=EN-US style='font-family:Kai'> manager</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td width=107 rowspan=3 style='width:80.2pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>group_manager</span></p>
  </td>
  <td width=246 colspan=4 style='width:184.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the group manager</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td width=132 colspan=2 rowspan=2 style='width:98.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>group_method</span></p>
  </td>
  <td width=114 colspan=2 style='width:85.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the group method</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td width=114 colspan=2 style='width:85.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=5 style='border:solid windowtext 1.0pt;border-top:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>queue_manager</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the </span><span
  lang=EN-US style='font-family:Kai'>queue manager</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>receiver</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the receiver</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>checker</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the checker</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=10 style='border:solid windowtext 1.0pt;border-top:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>client</span></p>
  </td>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>the path of the client</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>epochs</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>int</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>local epoch</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>batch_size</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>int</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>batch</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>model</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US style='font-family:Kai'>the path of the model</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>loss</span></p>
  </td>
  <td colspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'><a href="#adding-loss-function">explanation</a></span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>mu</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>float</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>proximal term’s coefficient</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>optimizer</span></p>
  </td>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>path</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>string</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>the path of the optimizer</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  <td colspan=3 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>dict</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>required parameters</span></p>
  </td>
 </tr>
 <tr style='height:1.0cm'>
  
  <td colspan=4 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>other_params</span></p>
  </td>
  <td width=47 style='width:34.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-family:Kai'>*</span></p>
  </td>
  <td width=496 style='width:371.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US style='font-family:Kai'>other parameters</span></p>
  </td>
 </tr>
 <tr height=0>
  <td width=107 style='border:none'></td>
  <td width=90 style='border:none'></td>
  <td width=42 style='border:none'></td>
  <td width=42 style='border:none'></td>
  <td width=72 style='border:none'></td>
  <td width=47 style='border:none'></td>
  <td width=496 style='border:none'></td>
 </tr>
</table>
</details>

## Adding New Algorithm

To allow clients/servers to call your own algorithms or implementation classes (note: all algorithm implementations must be in class form), the following steps are required:

* Add your own implementation to the corresponding location (dataset, model, schedule, update, client, loss)
* Import the class in the `__init__.py` file of the corresponding package, for example `from model import CNN`
* Declare in the configuration file, `model_path` corresponds to the path where the new algorithm is located.
* `checker`, `group`, `receiver`, `schedule`, and `update` modules need to be supplemented with invocation methods in the `Caller` class.
* `loss` and `numgenerator` modules need to be supplemented with invocation methods in the `factory` class.

In addition, parameters that the algorithm needs to use can be declared in the `params` configuration item.

Now the `model`, `optim`, and `loss` modules support the introduction of built-in implementation classes such as `torch`, for example:

```json
"model": {
      "path": "torchvision.models.resnet18",
      "params": {
        "pretrained": true,
        "num_classes": 10 
      }
}
```

### Adding Loss Function

The loss function is now generated and created by the `LossFactory` class. You can choose to use built-in algorithms from `Torch` or implement your own.

The loss configuration supports three settings. The first option is using a string format commonly used in the configuration file:

```json
"loss": "torch.nn.functional.cross_entropy"
```

In this case, the program will directly generate a loss function using the `functional` approach.

The second option is to generate an `object-based` loss:

```json
"loss": {
    "path": "loss.myloss.MyLoss",
    "params": {}
}
```

Here, you specify the path to your custom loss class and provide any necessary parameters in the params field.

The third option is to generate a loss based on the type:

```json
"loss": {
        "type": "func",
        "path": "loss.myloss.MyLoss",
        "params": {}
    }
```

With this option, you also provide the type field as "func", and the rest of the process is similar to the object-based approach.


## Staleness Settings

`stale` has three settings, one of which is mentioned in the above configuration file.

```json
"stale": {
      "step": 5,
      "shuffle": true,
      "list": [10, 10, 10, 5, 5, 5, 5]
    }
```

The program will generate a string of random integers based on the provided `step` and `list`. For example, in the code above, the program will generate 10 zeros, 10 (0, 5), and 10 [5, 10), and shuffle them if shuffle is set to true. Finally, the random string is assigned to each client, and the client sleeps according to the corresponding number of seconds after each round of training. When storing the JSON file to the experimental results, this setting will be automatically converted to the third setting.

The second option is to set it to false, in which case the program will set the delay for each client to 0.

```json
"stale": false
```

The third option is a list of random integers, and the program will directly assign the delay settings from the list to the clients.

```json
"stale": [1, 2, 3, 1, 4]
```

## Data Distribution Settings

### iid

When `iid` is set to true (in fact, it is also the default when set to false), the data will be distributed to each client in an identical and independent way (iid).

```json
"iid": true
```

### dirichlet non-iid

When `customize` in iid is set to false or not set, the data will be distributed to each client in a Dirichlet distribution. 

Beta is the parameter of the Dirichlet distribution.

```json
"iid": {
    "customize": false,
    "beta": 0.5
}
```

or

```json
"iid": {
    "beta": 0.5
}
```

### customize non-iid

Customized non-iid settings are divided into two parts, one is for label non-iid setting and the other is for data quantity non-iid setting. Currently, only random generation is provided for data quantity, and personalized settings will be introduced in future versions.

When enabling the customized setting, you need to set `customize` to true and set `label` and `data` separately.

```json
"iid": {
    "customize": true
}
```

#### label distribution

Label setting is similar to staleness settings and supports three modes. The first one is mentioned in the configuration file.

```json
"label": {
    "step": 1,
    "list": [10, 10, 30]
}
```

The above configuration will generate 10 clients with 1 label data, 10 clients with 2 label data, and 30 clients with 3 label data.

If `step` is set to 2, the program will generate 10 clients with 1 label data, 10 clients with 3 label data, and 30 clients with 5 label data.

The second option is a two-dimensional array of random numbers, and the program will assign the array directly to the clients.

```json
"label": {
    "0": [1, 2, 3, 8],
    "1": [2, 4],
    "2": [4, 7],
    "3": [0, 2, 3, 6, 9],
    "4": [5]
}
```

The third option is a one-dimensional array, which represents the number of labels each client has, and the length of the array should be the same as the number of clients.

```json
"label": {
  "list": [4, 5, 10, 1, 2, 3, 4]
}
```

The above configuration sets the number of label data for each client: client 0 has 4 label data, client 1 has 5 label data, and so on.

Currently, there are two randomization methods for generating label non-iid data, one is pure randomization, which may lead to all clients missing one label, resulting in a decrease in accuracy (although the probability is extremely low). The other method uses shuffle algorithm to ensure that each label is selected, but it also leads to the inability to generate data with uneven label distributions. The shuffle algorithm is controlled by the shuffle parameter, as shown below:

```json
"label": {
  "shuffle": true,
  "list": [4, 5, 10, 1, 2, 3, 4]
}
```

### data distribution

The data setting is relatively simple, currently there are two methods, one of which is empty.

```json
"data": {}
```

That is, no non-iid setting is performed on the data quantity.

The second method is mentioned in the configuration file.

```json
"data": {
    "max": 500,
    "min": 400
}
```

That is, the data quantity for each client will be randomly distributed between 400 and 500, and will be evenly distributed among the labels by default.

The data quantity distribution is still relatively simple at this point, and will be gradually improved in the future.

## Adding New Client Class

Currently, client replacement needs to inherit from `AsyncClient` or `SyncClient`, and the new parameters are passed into the class through the `client` configuration item.

## Multi-GPU

The multi-GPU feature of this project is not about multi-GPU parallel computing. Each client is still trained on a single GPU, but macroscopically, the clients run on multiple GPUs. That is, the training tasks of each client will be evenly distributed to `the GPUs visible to the program`. The GPU bound to each client is specified at initialization and is not specified on each round of training. Therefore, it is still possible to have a serious imbalance in GPU load.

This feature is controlled by the `multi_gpu` switch in the global settings.

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
    </td></tr>
</table>
<!-- readme: contributors -end -->

## Contact us

QQ: 527707607

email: desperado@qq.com

Welcome to provide suggestions for the project~

if you'd like contribute to this project, please contact us.
