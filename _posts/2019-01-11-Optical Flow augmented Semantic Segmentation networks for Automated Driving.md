---
layout:            post
title:             "Optical Flow augmented Semantic Segmentation networks for Automated Driving"
menutitle:         "Optical Flow augmented Semantic Segmentation networks for Automated Driving"
category:          Publications
author:            asallab
tags:              
---

[Rashed, Hazem, Senthil Yogamani, Ahmad El-Sallab, Pavel Krizek, and Mohamed El- Helw. "Optical Flow augmented Semantic Segmentation networks for Automated Driving." Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, Prague, Czech Republic, Feb 25-27, 2019](https://arxiv.org/abs/1901.07355)

 Motion is a dominant cue in automated driving systems. Optical flow is typically computed to detect moving objects and to estimate depth using triangulation. In this paper, our motivation is to leverage the existing dense optical flow to improve the performance of semantic segmentation. To provide a systematic study, we construct four different architectures which use RGB only, flow only, RGBF concatenated and two-stream RGB + flow. We evaluate these networks on two automotive datasets namely Virtual KITTI and Cityscapes using the state-of-the-art flow estimator FlowNet v2. We also make use of the ground truth optical flow in Virtual KITTI to serve as an ideal estimator and a standard Farneback optical flow algorithm to study the effect of noise. Using the flow ground truth in Virtual KITTI, two-stream architecture achieves the best results with an improvement of 4% IoU. As expected, there is a large improvement for moving objects like trucks, vans and cars with 38%, 28% and 6% increase in IoU. FlowNet produces an improvement of 2.4% in average IoU with larger improvement in the moving objects corresponding to 26%, 11% and 5% in trucks, vans and cars. In Cityscapes, flow augmentation provided an improvement for moving objects like motorcycle and train with an increase of 17% and 7% in IoU.