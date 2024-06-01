# SA-DRO-For-Fair-Supervised-Learning

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [References](#reference)

## Introduction

This project is an official implementation of the SA-DRO method in this paper, [**"On the Inductive Biases of Demographic Parity-based Fair Learning Algorithms"**](https://arxiv.org/abs/2402.18129), was accepted by UAI2024. 

This project includes the demo results for COMPAS and Adult datasets. The datasets are separated highly imbalanced to observe the inductive biases brought by fair classification algorithms, and users can tune the hyperparameters for fairness and distributional robustness to get different results for Accuracy, DDP, and Negative Rates.

This code is based on [KDE Method](https://proceedings.neurips.cc/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf) by Cho, J., Hwang, G., & Suh, C. [2020].

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/lh218/SA-DRO-For-Fair-Supervised-Learning

# Navigate to the project directory
cd SA-DRO-For-Fair-Supervised-Learning

```

## Usage

```bash

# Example of usage
python trainer_COMPAS.py

python trainer_Adult.py

```

## References

1. - Cho, Jaewoong, Gyeongjo Hwang, and Changho Suh. "A fair classifier using kernel density estimation." Advances in neural information processing systems 33 (2020): 15088-15099. [KDE](https://proceedings.neurips.cc/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf)
