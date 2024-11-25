# GarVerseLOD
This repository includes the related code of *GarVerseLOD*.

> **GarVerseLOD**: High-Fidelity 3D Garment Reconstruction from a Single In-the-Wild Image using a Dataset with Levels of Details
>
> [Zhongjin Luo](https://zhongjinluo.github.io/), [Haolin Liu](https://haolinliu97.github.io/), [Chenghong Li](https://kevinlee09.github.io/), Wanghao Du, Zirong Jin, Wanhu Sun, [Yinyu Nie](https://yinyunie.github.io/), [Weikai Chen](https://chenweikai.github.io/), [Xiaoguang Han](https://gaplab.cuhk.edu.cn/)

#### | [Paper](https://arxiv.org/abs/2411.03047) | [Project](https://garverselod.github.io/) |

## Introduction

![gallery](./assets/fig_teaser.png)

We propose a hierarchical framework to recover different levels of garment details by leveraging the garment shape and deformation priors from the GarVerseLOD dataset. Given a single clothed human image searched from Internet, our approach is capable of generating high-fidelity 3D standalone garment meshes that exhibit realistic deformation and are well-aligned with the input image.

## Install

```
git clone https://github.com/zhongjinluo/GarVerseLOD.git
cd GarVerseLOD
conda env create -f environment.yml
conda activate garverselod
```

This system has been tested with Python 3.8.19, PyTorch 1.13.1, PyTorch3D 0.7.1 and CUDA 11.7 on Ubuntu 18.04.

## Demo
To run our system, please refer to [demo/README.md](demo/) for instructions.