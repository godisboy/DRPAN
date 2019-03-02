# DRPAN: Discriminative Region Proposal Adversarial Network for High-Quality Image-to-Image Translation

This is our implementation of DRPAN.

## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/

- Clone this repo:
```bash
git clone https://github.com/godisboy/DRPAN.git
cd DRPAN
```
- Prepare your paired image2image translation dataset:

- Modify the config file:
change the `dataPath` to your data set path

### DRPAN train
- Train a model:
```
python main.py --config configs/facades.yaml --cuda --gpu_ids 0

```
### StackGan-like model train
```
python train_stack_pix2pix.py --config configs/facades.yaml --cuda --gpu_ids 0

```
## Paper

Chao Wang, Haiyong Zheng, Zhibin Yu, Ziqiang Zheng, Zhaorui Gu, Bing Zheng. "Discriminative Region Proposal Adversarial Networks for High-Quality Image-to-Image Translation", ECCV 2018. [[CVF](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chao_Wang_Discriminative_Region_Proposal_ECCV_2018_paper.pdf)] [[arXiv](https://arxiv.org/abs/1711.09554)]

## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2).
