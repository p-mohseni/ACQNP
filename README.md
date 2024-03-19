Adaptive Conditional Quantile Neural Processes
============

This is a PyTorch implementation of the A/CQNP and competitor baselines as described in our paper [Adaptive Conditional Quantile Neural Processes](https://arxiv.org/abs/2305.18777) appeared in 39-th Conference on Uncertainty in Artificial Intelligence (UAI 2023).

## Abstract
Neural processes are a family of probabilistic models that inherit the flexibility of neural networks to parameterize stochastic processes. Despite providing well-calibrated predictions, especially in regression problems, and quick adaptation to new tasks, the Gaussian assumption that is commonly used to represent the predictive likelihood fails to capture more complicated distributions such as multimodal ones. To overcome this limitation, we propose Conditional Quantile Neural Processes (CQNPs), a new member of the neural processes family, which exploits the attractive properties of quantile regression in modeling the distributions irrespective of their form. By introducing an extension of quantile regression where the model learns to focus on estimating informative quantiles, we show that the sampling efficiency and prediction accuracy can be further enhanced. Our experiments with real and synthetic datasets demonstrate substantial improvements in predictive performance compared to the baselines, and better modeling of heterogeneous distributions' characteristics such as multimodality.

## Requirements
We have tested our implementation on Python 3 with the following packages:

PyTorch = 1.9.0  
torchvision = 0.10.0  
CUDA = 11.1.1

## Cite

If you find this work useful in your research, please consider citing our paper:

```
@inproceedings{mohseni2023adaptive,
  title={Adaptive conditional quantile neural processes},
  author={Mohseni, Peiman and Duffield, Nick and Mallick, Bani and Hasanzadeh, Arman},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1445--1455},
  year={2023},
  organization={PMLR}
}
```
