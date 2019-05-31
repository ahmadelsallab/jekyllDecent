---
layout:            post
title:             "Meta learning Framework for Automated Driving"
menutitle:         "Meta learning Framework for Automated Driving"
category:          Publications
author:            asallab
tags:              
---

[Ahmad A. Al Sallab, Mahmoud Saeed, Omar AbdelTawab, Mohammed Abdo. “Meta learning Framework for Automated Driving”, International Conference on Machine Learning (ICML), Machine Learning in Automated Driving workshop, 4-10 August 2017, Syndey, Australia.](https://arxiv.org/abs/1706.04038)


The success of automated driving deployment is highly depending on the ability to develop an efficient and safe driving policy. The problem is well formulated under the framework of optimal control as a cost optimization problem. Model based solutions using traditional planning are efficient, but require the knowledge of the environment model. On the other hand, model free solutions suffer sample inefficiency and require too many interactions with the environment, which is infeasible in practice. Methods under the Reinforcement Learning framework usually require the notion of a reward function, which is not available in the real world. Imitation learning helps in improving sample efficiency by introducing prior knowledge obtained from the demonstrated behavior, on the risk of exact behavior cloning without generalizing to unseen environments. In this paper we propose a Meta learning framework, based on data set aggregation, to improve generalization of imitation learning algorithms. Under the proposed framework, we propose MetaDAgger, a novel algorithm which tackles the generalization issues in traditional imitation learning. We use The Open Race Car Simulator (TORCS) to test our algorithm. Results on unseen test tracks show significant improvement over traditional imitation learning algorithms, improving the learning time and sample efficiency in the same time. The results are also supported by visualization of the learnt features to prove generalization of the captured details.