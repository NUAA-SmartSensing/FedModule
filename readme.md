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
│   ├── pic
│   │   ├── fedsemi.png
│   │   ├── framework.png
│   │   └── header.png
│   └── readme-zh.md
├── docker
│   └── Dockerfile
├── license
├── readme.md
├── requirements.txt
└── src 
    ├── client                                Client implementation
    │   ├── ActiveClient.py
    │   ├── Client.py
    │   ├── DLClient.py
    │   ├── NormalClient.py
    │   ├── ProxClient.py
    │   ├── SemiClient.py
    │   ├── TestClient.py
    │   └── __init__.py
    ├── data                                  Dataset download location
    ├── dataset                               Dataset class
    │   ├── CIFAR10.py
    │   ├── MNIST.py
    │   ├── FashionMNIST.py
    │   └── __init__.py
    ├── exception                             Exception class
    │   ├── ClientSumError.py
    │   └── __init__.py
    ├── fedasync                              Asynchronous Federated Learning
    │   ├── AsyncClientManager.py
    │   ├── AsyncServer.py
    │   ├── QueueManager.py
    │   ├── SchedulerThread.py
    │   ├── UpdaterThread.py
    │   ├── __init__.py
    │   ├── checker
    │   │   └── AllChecker.py
    │   ├── quantitydeterminer
    │   └── receiver
    │       └──  AvgReceiver.py
    ├── fedsemi                               Semi-asynchronous Federated Learning
    │   ├── QueueManager.py                   Message Queue Manager class
    │   ├── SchedulerThread.py                Scheduling Thread
    │   ├── SemiAsyncClientManager.py         Client Manager class
    │   ├── SemiAsyncServer.py                Server class
    │   ├── UpdaterThread.py                  Aggregation Thread
    │   ├── __init__.py
    │   ├── checker                           Semi-asynchronous checker
    │   │   └── SemiAvgChecker.py
    │   ├── grouping                          Partitioner
    │   │   ├── Grouping.py
    │   │   ├── NormalGrouping.py
    │   │   └── SimpleGrouping.py
    │   └── receiver                          Semi-asynchronous receiver
    │       └── SemiAvgReceiver.py
    ├── fedsync                               Synchronous Federated Learning
    │   ├── QueueManager.py                   Queue Manager class
    │   ├── SchedulerThread.py                Scheduling Thread
    │   ├── SyncClientManager.py              Client Manager class
    │   ├── SyncServer.py
    │   ├── UpdaterThread.py
    │   ├── __init__.py
    │   ├── checker                           Synchronous Checker
    │   │   └── AvgChecker.py
    │   └── receiver                          Synchronous Receiver
    │       └── AvgReceiver.py
    ├── fl                                    fl main function
    │   ├── __init__.py
    │   ├── main.py
    │   └── wandb                             wandb running directory
    ├── loss                                  Implementation of Loss Function
    │   ├── FedLC.py
    │   └── __init__.py
    ├── model
    │   ├── CNN.py
    │   └── __init__.py
    ├── results
    ├── schedule                              Scheduling Algorithm Class
    │   ├── FullSchedule.py
    │   ├── NoSchedule.py
    │   ├── RandomSchedule.py
    │   ├── RoundRobin.py
    │   └── __init__.py
    ├── test                                  for test
    ├── update                                Updating Algorithm Class
    │   ├── AsyncAvg.py
    │   ├── FedAT.py
    │   ├── FedAsync.py
    │   ├── FedAvg.py
    │   ├── FedDL.py
    │   ├── StepAsyncAvg.py
    │   └── __init__.py
    └── utils
        ├── ConfigManager.py
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

### Asynchronous Configuration

[example](config/FedAsync-config.json)

### Synchronous Configuration

[example](config/FedAvg-config.json)

### Semi-aynchronous Configuration

[example](config/FedAT-config.json)

### Parameter explanation

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td width=330 colspan=9 style='width:247.65pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'>parameters</p>
  </td>
  <td width=103 style='width:76.95pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'>type</p>
  </td>
  <td width=120 style='width:90.2pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'>explanations</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:2.75pt'>
  <td width=106 colspan=2 rowspan=3 style='width:79.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>wandb</span></span></p>
  </td>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>enabled</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US>bool</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  class=SpellE><span lang=EN-US>whether to enable wandb</span></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:2.65pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>project</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'>project name</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:2.65pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>name</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.65pt'>
  <p class=MsoNormal align=center style='text-align:center'>the name of this run</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:1.05pt'>
  <td width=106 colspan=2 rowspan=8 style='width:79.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>global</span></p>
  </td>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>use_file_system</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>bool</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.05pt'>
  <p class=MsoNormal align=center style='text-align:center'>whether to enable the file system as the torch multi-thread sharing strategy></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>multi_gpu</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>bool</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US>whether to enable multi-GPU, </span><a href="#multi-gpu">detailed explanation</a></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>mode</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  class=GramE><span lang=EN-US>async,sync</span></span></span><span lang=EN-US>,
  semi-async</span></p>
  <p class=MsoNormal align=center style='text-align:center'>choose one of the three operating modes of the framework</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>experiment</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'>the name of this run</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>stale</span></p>
  </td>
  <td width=223 colspan=2 style='width:167.15pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><a href="#staleness-settings">explaination</a></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>dataset_path</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the dataset</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>iid</span></span></p>
  </td>
  <td width=223 colspan=2 style='width:167.15pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><a href="#iid">explaination</a></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:1.0pt'>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>client_num</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>int</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'>client num</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:2.05pt'>
  <td width=106 colspan=2 rowspan=12 style='width:79.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>server</span></p>
  </td>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>epochs</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>int</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.05pt'>
  <p class=MsoNormal align=center style='text-align:center'>global epoch</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:3.55pt'>
  <td width=99 colspan=4 rowspan=2 style='width:74.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987755'><span lang=EN-US>model</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987755'></span>
  <td width=125 colspan=3 style='width:93.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987755'><span lang=EN-US>path</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987755'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987755'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987755'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987755'>the path of the model</span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987755'></span>
 </tr>
 <tr style='mso-yfti-irow:14;height:3.5pt'>
  <td width=125 colspan=3 style='width:93.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>params</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  class=SpellE><span style='mso-bookmark:OLE_LINK8'><span lang=EN-US>dict</span></span></span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'>required parameters</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15;height:2.35pt'>
  <td width=72 colspan=2 rowspan=4 style='width:54.15pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>scheduler</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=152 colspan=5 style='width:114.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>scheduler_path</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the scheduler</p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:16;height:2.35pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=152 colspan=5 style='width:114.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>params</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:2.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'>required parameters</span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:17;height:3.55pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=65 colspan=4 rowspan=2 style='width:48.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the receiver</p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=88 style='width:65.7pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>path</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the receiver</p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:18;height:3.5pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=88 style='width:65.7pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>params</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>dict</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'>required parameters</span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:19;height:1.8pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=62 rowspan=5 style='width:46.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.8pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>updater</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=162 colspan=6 style='width:121.85pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.8pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>updater_path</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.8pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.8pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the updater</p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:20;height:1.75pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=162 colspan=6 style='width:121.85pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span style='mso-bookmark:_Hlk132988020'><span
  lang=EN-US>loss</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'><span style='mso-bookmark:_Hlk132988020'></span></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span style='mso-bookmark:_Hlk132988020'><span
  class=SpellE><span lang=EN-US>dict</span></span><span lang=EN-US> | string</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'><span style='mso-bookmark:_Hlk132988020'></span></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'>the parameters required by customized loss | the path of the loss</p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'><span style='mso-bookmark:_Hlk132988020'></span></span>
 </tr>
 <tr style='mso-yfti-irow:21;height:1.75pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=162 colspan=6 style='width:121.85pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>params</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>dict</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'>required parameters</span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:22;height:3.55pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=68 colspan=4 rowspan=2 style='width:50.9pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>group[exclusive to semi-async]</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=95 colspan=2 style='width:70.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>updater_path</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>string</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'><span
  style='mso-bookmark:_Hlk132987069'></span>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>exclusive to sam</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:23;height:3.5pt'>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=95 colspan=2 style='width:70.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span lang=EN-US>params</span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'><span class=SpellE><span lang=EN-US>dict</span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='mso-bookmark:_Hlk132987069'>required parameters</span></p>
  </td>
  <span style='mso-bookmark:_Hlk132987069'></span>
 </tr>
 <tr style='mso-yfti-irow:24;height:4.05pt'>
  <td width=106 colspan=2 rowspan=3 style='width:79.25pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:4.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>client_manager</span></span></p>
  </td>
  <td width=225 colspan=7 style='width:168.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:4.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>client_path</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:4.05pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:4.05pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the client</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:25;height:3.55pt'>
  <td width=99 colspan=4 rowspan=2 style='width:74.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>checker</span></p>
  </td>
  <td width=125 colspan=3 style='width:93.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>checker_path</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the checker</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:26;height:3.5pt'>
  <td width=125 colspan=3 style='width:93.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>params</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>dict</span></span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'>required parameters</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:27;height:1.55pt'>
  <td width=104 rowspan=9 style='width:77.75pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>client</span></p>
  </td>
  <td width=227 colspan=8 style='width:169.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>epochs</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>int</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>local epoch</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:28;height:1.3pt'>
  <td width=227 colspan=8 style='width:169.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>batch_size</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>int</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>batch</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:29;height:3.55pt'>
  <td width=100 colspan=4 rowspan=2 style='width:75.25pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>model</span></p>
  </td>
  <td width=126 colspan=4 style='width:94.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>path</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the model</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:30;height:3.5pt'>
  <td width=126 colspan=4 style='width:94.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>params</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>dict</span></span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'>required parameters</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:31;height:1.3pt'>
  <td width=227 colspan=8 style='width:169.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>loss</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>dict</span></span><span lang=EN-US> | string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'>the parameters required by customized loss | the path of the loss</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:32;height:1.3pt'>
  <td width=227 colspan=8 style='width:169.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>mu</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>float</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.3pt'>
  <p class=MsoNormal align=center style='text-align:center'>proximal term’s coefficient</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:33;height:3.55pt'>
  <td width=100 colspan=4 rowspan=2 style='width:75.25pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>optimizer</span></p>
  </td>
  <td width=126 colspan=4 style='width:94.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>path</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>string</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.55pt'>
  <p class=MsoNormal align=center style='text-align:center'>the path of the optimizer</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:34;height:3.5pt'>
  <td width=126 colspan=4 style='width:94.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>params</span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>dict</span></span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'>required parameters</p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:35;mso-yfti-lastrow:yes;height:3.5pt'>
  <td width=227 colspan=8 style='width:169.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US>other_params</span></span></p>
  </td>
  <td width=103 style='width:76.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>*</span></p>
  </td>
  <td width=120 style='width:90.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:3.5pt'>
  <p class=MsoNormal align=center style='text-align:center'>other parameters</p>
  </td>
 </tr>
</table>

## Adding New Algorithm

To allow clients/servers to call your own algorithms or implementation classes (note: all algorithm implementations must be in class form), the following steps are required:

* Add your own implementation to the corresponding location (dataset, model, schedule, update, client, loss)
* Import the class in the `__init__.py` file of the corresponding package, for example `from model import CNN`
* Declare in the configuration file, `model_path` corresponds to the path where the new algorithm is located.

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

The loss function can use the built-in algorithms in PyTorch, or it can be implemented separately. The steps for separate implementation are mostly the same as above. The following modifications need to be made in the configuration item:

```json
"client": {
    "loss": {
        "loss_file": "my_loss",
        "loss_name": "my_loss"
  }
}
```

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
