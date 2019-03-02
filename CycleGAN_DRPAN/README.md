# CycleGAN+DRPAN in PyTorch

This is our PyTorch implementation of CycleGAN+DRPAN on unpaired image-to-image translation.

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Install PyTorch 0.4+ and torchvision from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/godisboy/DRPAN.git
cd DRPAN/CycleGAN_DRPAN
```
### CycleGAN+DRPAN train/test

- Download a CycleGAN dataset (e.g. cityscapes):
```bash
bash ./datasets/download_cyclegan_dataset.sh cityscapes
```
- Train a model:
```bash
python train.py --dataroot ./datasets/cityscapes --name cityscapes_cycle_drpan --gpu_ids 0
```
The train results will be saved to `./checkpoints/cityscapes_cycle_drpan`.

- Test the model:
```bash
python test.py --dataroot ./datasets/cityscapes --name cityscapes_cycle_drpan --gpu_ids 0
```
The test results will be saved to `./results/cityscapes_cycle_drpan`.

