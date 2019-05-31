---
layout:            post
title:             "YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds."
menutitle:         "YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds."
category:          Publications
author:            asallab
tags:              
---

[Ahmad A. Al Sallab, Ibrahim Sobh, Mahmoud Zidan, Mohamed Zahran and Sherif Abdelkarim. “YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds.”, Neural Information Processing Systems (NIPS), Machine Learning in Intelligent Transportation MLITS workshop, 05 Dec – 10 Dec, 2018, Canada](https://openreview.net/forum?id=B1xWZic29m)

 In this paper, YOLO4D is presented for Spatio-temporal Real-time 3D Multi-object detection and classification from LiDAR point clouds. Automated Driving dynamic scenarios are rich in temporal information. Most of the current 3D Object Detection approaches are focused on processing the spatial sensory features, either in 2D or 3D spaces, while the temporal factor is not fully exploited yet, especially from 3D LiDAR point clouds. In YOLO4D approach, the 3D LiDAR point clouds are aggregated over time as a 4D tensor; 3D space dimensions in addition to the time dimension, which is fed to a one-shot fully convolutional detector, based on YOLO v2. The outputs are the oriented 3D Object Bounding Box information, together with the object class. Two different techniques are evaluated to incorporate the temporal dimension; recurrence and frame stacking. The experiments conducted on KITTI dataset, show the advantages of incorporating the temporal dimension.