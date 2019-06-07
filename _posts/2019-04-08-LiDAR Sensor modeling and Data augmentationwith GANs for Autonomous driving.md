---
layout:            post
title:             "LiDAR Sensor modelling and Data augmentation with GANs for Autonomous driving"
menutitle:         "LiDAR Sensor modelling and Data augmentation with GANs for Autonomous driving"
category:          Publications
author:            asallab
tags:              
---

[Ahmad El Sallab, Ibrahim Sobh, Mohamed Zahran, Nader Essam, “LiDAR Sensor modelling and Data augmentation with GANs for Autonomous driving”, Proceedings of the 36th International Conference on MachineLearning (ICML), 2019, Long Beach, California.](https://arxiv.org/abs/1905.07290)

 In the autonomous driving domain, data collection and annotation from real vehicles are expensive and sometimes unsafe. Simulators are often used for data augmentation, which requires realistic sensor models that are hard to formulate and model in closed forms. Instead, sensors models can be learned from real data. The main challenge is the absence of paired data set, which makes traditional supervised learning techniques not suitable. In this work, we formulate the problem as image translation from unpaired data and employ CycleGANs to solve the sensor modeling problem for LiDAR, to produce realistic LiDAR from simulated LiDAR (sim2real). Further, we generate high-resolution, realistic LiDAR from lower resolution one (real2real). The LiDAR 3D point cloud is processed in Bird-eye View and Polar 2D representations. The experimental results show a high potential of the proposed approach.
