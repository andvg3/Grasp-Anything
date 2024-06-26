# Grasp-Anything
This is the repository of the paper "Grasp-Anything: Large-scale Grasp Dataset from Foundation Models"
## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)

## Installation
- Create a virtual environment
```bash
$ conda create -n granything python=3.9
$ conda activate granything
```

- Install pytorch
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Datasets
Our dataset can be accessed via [this link](https://airvlab.github.io/grasp-anything/docs/download/).

## Training
We use GR-ConvNet as our default deep network. To train GR-ConvNet on different datasets, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0
```
For example, if you want to train a GR-ConvNet on Cornell, use the following command:
```bash
$ python train_network.py --dataset cornell --dataset-path data/cornell --description training_cornell --use-depth 0
```
We also provide training for other baselines, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0 --network <baseline_name>
```
For instance, if you want to train GG-CNN on Cornell, use the following command:
```bash
python train_network.py --dataset cornell --dataset-path data/cornell/ --description training_ggcnn_on_cornell --use-depth 0 --network ggcnn
```

## Testing
For testing procedure, we can apply the similar commands to test different baselines on different datasets:
```bash
python evaluate.py --network <path_to_pretrained_network> --dataset <dataset> --dataset-path data/<dataset> --iou-eval
```
Important note: `<path_to_pretrained_network>` is the path to the pretrained model obtained by training procedure. Usually, the pretrained models obtained by training are stored at `logs/<timstamp>_<training_description>`. You can select the desired pretrained model to evaluate. We do not have to specify neural architecture as the codebase will automatically detect the neural architecture. Pretrained weights are available at [this link](https://drive.google.com/file/d/1OXVFXqv0rgxiVLz89tnSj0Xb-20ZJ4fH/view?usp=sharing).


## Acknowledgement
Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
