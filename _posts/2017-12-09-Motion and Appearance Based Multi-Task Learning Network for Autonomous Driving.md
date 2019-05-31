---
layout:            post
title:             "Motion and Appearance Based Multi-Task Learning Network for Autonomous Driving"
menutitle:         "Motion and Appearance Based Multi-Task Learning Network for Autonomous Driving"
category:          Publications
author:            asallab
tags:              
---

[Siam, M., Mahgoub, H., Zahran, M., Yogamani, S., Jagersand, M. and Ahmad A. Al Sallab, “Motion and Appearance Based Multi-Task Learning Network for Autonomous Driving.”, Neural Information Processing Systems (NIPS), Machine Learning in Inetelligent Transportation MLITS workshop, 05 Dec – 10 Dec, 2017, Long Beach,
California, USA.](https://openreview.net/pdf?id=Bk4BBBLRZ)

Autonomous driving has various visual perception tasks such as object detection, motion detection, depth estimation and flow estimation. Multi-task learning (MTL) has been successfully used for jointly estimating some of these tasks. Previous work was focused on utilizing appearance cues. In this paper, we address the gap of incorporating motion cues in a multi-task learning system. We propose a novel two-stream architecture for joint learning of object detection, road segmentation and motion segmentation. We designed three different versions of our network to establish systematic comparison. We show that the joint training of tasks significantly improves accuracy compared to training them independently even with a relatively smaller amount of annotated samples for motion segmentation. To enable joint training, we extended KITTI object detection dataset to include moving/static annotations of the vehicles. An extension of this new dataset named KITTI MOD is made publicly available via the official KITTI benchmark [website](http://www.cvlibs.net/datasets/kitti/eval_semantics.php). Our baseline network outperforms MPNet which is a state of the art for single stream CNN-based motion detection. The proposed two-stream architecture improves the mAP score by 21.5% in KITTI MOD. We also evaluated our algorithm on the non-automotive DAVIS dataset and obtained accuracy close to the state-of-the-art performance. The proposed network runs at 8 fps on a Titan X GPU using a two-stream VGG16 encoder. Demonstration of the work is provided [here](https://www.youtube.com/watch?v=hwP_oQeULfc) 
