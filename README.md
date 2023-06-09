# reproduce MEEE-torch

## Overview

Paper: [Sample Efficient Reinforcement Learning via Model-Ensemble Exploration and Exploitation](https://arxiv.org/abs/2107.01825)

MEEE origin repo: https://github.com/YaoYao1995/MEEE (tensorflow)

We implement MEEE based on [mbpo-torch](https://github.com/Xingyu-Lin/mbpo_pytorch).

## MBPO-Overview

This is a re-implementation of the model-based RL algorithm MBPO in pytorch as described in the following paper: [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253).

This code is based on a [previous paper in the NeurIPS reproducibility challenge](https://openreview.net/forum?id=rkezvT9f6r) that reproduces the result with a tensorflow ensemble model but shows a significant drop in performance with a pytorch ensemble model. 
This code re-implements the ensemble dynamics model with pytorch and closes the gap. 

## MBPO-Reproduced results

The comparison are done on two tasks while other tasks are not tested. But on the tested two tasks, the pytorch implementation achieves similar performance compared to the official tensorflow code.
![alt text](./results/hopper.png) ![alt text](./results/walker2d.png)

## Dependencies

MuJoCo 1.5 & MuJoCo 2.0

## Usage

> CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --env_name 'Ant-v2' --num_epoch 300 --model_type 'pytorch' --exp_name ant --num_networks 2 --num_elites 2 --seed 1
>  
> CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --env_name 'InvertedPendulum-v2' --num_epoch 400 --exp_name inver_mbpo_0

> my launch:
>
> python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 120
>
> python main_mbpo_dmc.py --domain_name 'cartpole' --task_name 'swingup' --num_epoch 110 --exp_name 'cartpole-swingup-1'

> for debug:
>
> python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 150 --exp_name 'hopper-1' --replay_size 1000
>
> python main_mbpo_dmc.py --domain_name 'cartpole' --task_name 'swingup' --num_epoch 110 --exp_name 'cartpole-swingup-1' --replay_size 1000

## Reference

* Official tensorflow implementation: https://github.com/JannerM/mbpo
* Code to the reproducibility challenge paper: https://github.com/jxu43/replication-mbpo
