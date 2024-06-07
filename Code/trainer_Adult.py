import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
import torch.optim as optim

from models import Classifier
from dataloader import FairnessDataset
from algorithm import train_fair_classifier

import warnings
warnings.simplefilter('ignore')

# Choose which dataset to load
dataset_names = ['Adult'] #COMPAS or Adult

# Model architecture
n_layers = 4 # MLP-hidden layer
n_hidden_units = 512 # MLP-hidden node

# KDE method hyperparameters
h = 0.1 # Bandwidth hyperparameter in KDE [positive real numbers]
delta = 1.0 # Delta parameter in Huber loss [positive real numbers]
lambda_ = 0.95 # Fairness regularization coefficient from [0, 1]

# DRO hyperparameter
reg_strength = 0.9 # Distributional robustness regularization coefficient from [0, +inf]

# Training hyperparameters
batch_size = 2000
lr = 1e-5
lr_decay = 1.0 # Exponentiacd l decay factor of LR scheduler
n_epochs = 500

device = torch.device('cuda') # or torch.device('cpu')

print_log = open("log_Adult.txt","w")
sys.stdout = print_log

result = pd.DataFrame()
starting_time = time.time()

for dataset_name in dataset_names:
    seed = 0
    print('Currently working on - Fairness lambda:', lambda_)
    print('Currently working on - Robustness sigma:', reg_strength)

    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]

    # Create a classifier model
    net = Classifier(n_layers=n_layers, n_inputs=input_dim, n_hidden_units=n_hidden_units)
    net = net.to(device)

    # Set an optimizer
    optimizer_m = optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer_m, gamma=lr_decay) # None

    # Fair classifier training
    temp, temp_z0, temp_z1 = train_fair_classifier(dataset=dataset, 
                                 net=net, optimizer_m=optimizer_m, lr_scheduler=lr_scheduler,
                                 lambda_=lambda_, h=h, delta=delta, 
                                 device=device, n_epochs=n_epochs, batch_size=batch_size, reg_strength=reg_strength)
    
    result = result.append(temp)
    result = result.append(temp_z0)
    result = result.append(temp_z1)
    
print('Average running time: {:.3f}s'.format((time.time() - starting_time)))
print(result)

print_log.close()