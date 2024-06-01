# SA-DRO For Fair Inductive Biases

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [References](#reference)

## Introduction

This project includes the implementation of the SA-DRO method in the paper, [**"On the Inductive Biases of Demographic Parity-based Fair Learning Algorithms"**](https://arxiv.org/abs/2402.18129), UAI2024. 

This project includes the demo results for COMPAS and Adult datasets. The datasets are separated into groups with an imbalanced sensitive attribute distribution to examine the inductive biases of fair learning algorithms.

The shared code follows the fair learning algorithms: [KDE Method](https://proceedings.neurips.cc/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf) by Cho, J., Hwang, G., & Suh, C. [2020].

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/lh218/Fairness-IB

# Navigate to the project directory
cd Fairness-IB

```

## Usage

```bash

# Example of usage
python trainer_COMPAS.py

python trainer_Adult.py

```

## References

1. Cho, Jaewoong, Gyeongjo Hwang, and Changho Suh. "A fair classifier using kernel density estimation." Advances in neural information processing systems 33 (2020): 15088-15099. [https://proceedings.neurips.cc/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf)
