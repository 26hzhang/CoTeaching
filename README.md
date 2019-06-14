# CoTeaching

This repository reproduces the NeurIPS'18 paper
[Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](
https://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels.pdf) 
by TensorFlow.

- TensorFlow implementation, see all `*_tf.py` files.
- Adapt original co-teaching PyTorch implementation to PyTorch `1.1.0`, see all `*_th.py` files. The original PyTorch
implementation is provided by the author "[Bo Han](https://bhanml.github.io)" as: [[bhanML/Co-teaching]](
https://github.com/bhanML/Co-teaching).

## Requirements
The codes are developed and tested on MacOS (`python==3.7.x`, CPU) and Ubuntu 18.04 (`python==3.6.x`, NVIDIA GeForce GTX 
1080 Ti GPU with `CUDA==10.0`) with following environment:
- tensorflow==1.13.1 (>=1.8.0)
- pytorch==1.1.0 (>=0.4.1)
- numpy==1.14.6 (>=1.14.2)

## Setups
**On MacOS**

Install TensorFlow via:
```bash
$ pip3 install tensorflow==1.13.1
```
Install PyTorch via:
```bash
$ pip3 install torch torchvision
```
**On Ubuntu**

Install TensorFlow via:
```bash
$ pip3 install tensorflow==1.13.1  # CPU version
$ pip3 install tensorflow-gpu==1.13.1  # GPU version
```
Install PyTorch via:
```bash
# CPU version
$ pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl  
$ pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
# GPU version
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
$ pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

## Usage
Here is an example for TensorFlow:
```bash
$ python3 main_tf.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5
```
Here is an example for PyTorch: 
```bash
$ python3 main_th.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5
```

## Performance
Performance on benchmark datasets reported by the Author:

| (Flipping, Rate) | MNIST  | CIFAR-10 | CIFAR-100 |
| ---------------: | -----: | -------: | --------: |
| (Pair, 45%)      | 87.58% | 72.85%   | 34.40%    |
| (Symmetry, 50%)  | 91.68% | 74.49%   | 41.23%    |
| (Symmetry, 20%)  | 97.71% | 82.18%   | 54.36%    |

Performance on benchmark datasets derived by the codes in this repository:
> `th` means PyTorch while `tf` means TensorFlow. 

| (Flipping, Rate) | MNIST (th -- tf) | CIFAR-10 (th -- tf) | CIFAR-100 (th -- tf) |
| ---------------: | ---------------: | ------------------: | -------------------: |
| (Pair, 45%)      | 88.63% -- 94.16% | 72.88% -- 76.04%    | 34.05% -- 35.24%     |
| (Symmetry, 50%)  | 92.34% -- 98.05% | 74.56% -- 79.64%    | 41.17% -- 49.09%     |
| (Symmetry, 20%)  | 97.84% -- 99.16% | 82.87% -- 87.02%    | 54.11% -- 59.55%     |

> The model structure and parameters setting of TensorFlow version are almost same as those of PyTorch version, 
but the performance of TensorFlow version is generally better than the PyTorch version, I think it maybe caused by the 
internal implementation of some functions are different between these two frameworks?
