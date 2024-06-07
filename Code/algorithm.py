import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from utils import measures_from_Yhat

tau = 0.5
k = 10

a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

import torch
import torch.nn as nn
import torch.nn.functional as F

def CDF_tau(Yhat, h=0.01, tau=0.5):
    m = len(Yhat)
    Y_tilde = (tau-Yhat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def train_fair_classifier(dataset, net, optimizer_m, lr_scheduler, lambda_, h, delta, device, n_epochs, batch_size, reg_strength):
    
    # Load datasets
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors

    X_test, Y_test, Z_test, XZ_test = test_tensors
    X_test_z0, Y_test_z0, Z_test_z0, XZ_test_z0 = X_test[Z_test==0], Y_test[Z_test==0], Z_test[Z_test==0], XZ_test[Z_test==0]
    X_test_z1, Y_test_z1, Z_test_z1, XZ_test_z1 = X_test[Z_test==1], Y_test[Z_test==1], Z_test[Z_test==1], XZ_test[Z_test==1]
    
    # Original distributions
    print("NR(Z = 0): %.2f%%" % (float((len(Y_test_z0)-sum(Y_test_z0))/len(Y_test_z0)*100)))
    print("NR(Z = 1): %.2f%%" % (float((len(Y_test_z1)-sum(Y_test_z1))/len(Y_test_z1)*100)))
    print("Train data: %d, Test data : %d" % (len(X_train), len(X_test)))
    print("Z = 1 train: %d, Z = 0 train: %d" % (sum(Z_train==1), len(Z_train)-sum(Z_train==1)))
    print("Z = 1 test: %d, Z = 0 test: %d" % (sum(Z_test==1), len(Z_test)-sum(Z_test==1)))
    
    # Retrieve train/test splitted numpy arrays for index=split
    train_arrays, test_arrays = dataset.get_dataset_in_ndarray()
    X_train_np, Y_train_np, Z_train_np, XZ_train_np = train_arrays
    
    X_test_np, Y_test_np, Z_test_np, XZ_test_np = test_arrays
    Y_test_np_z0, Y_test_np_z1 = Y_test_np[Z_test_np==0], Y_test_np[Z_test_np==1]
    Z_test_np_z0, Z_test_np_z1 = Z_test_np[Z_test_np==0], Z_test_np[Z_test_np==1]
    
    # Obtain the marginal distribution for sensitive attributes
    
    q_z0 = torch.tensor(float(sum(Z_train_np==0) / len(Z_train_np))).to(device)
    q_z1 = torch.tensor(float(sum(Z_train_np==1) / len(Z_train_np))).to(device)
    
    sensitive_attrs = dataset.sensitive_attrs

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi)
    
    df_ckpt = pd.DataFrame()
    
    loss_function = nn.BCELoss()
    total_loss = 0
    costs = []
    
    # Set the inner loop optimizer
    p1 = torch.tensor([0.5])
    p1 = p1.to(device)
    p1.requires_grad = True
    optimizer_p = optim.SGD([p1], lr=2e-3)
    
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
             
            # Batch selection by sensitive attributes
            xz_batch_z0, y_batch_z0, z_batch_z0 = xz_batch[z_batch==0], y_batch[z_batch==0], z_batch[z_batch==0]
            xz_batch_z1, y_batch_z1, z_batch_z1 = xz_batch[z_batch==1], y_batch[z_batch==1], z_batch[z_batch==1]
            
            xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
            xz_batch_z0, y_batch_z0, z_batch_z0 = xz_batch_z0.to(device), y_batch_z0.to(device), z_batch_z0.to(device)
            xz_batch_z1, y_batch_z1, z_batch_z1 = xz_batch_z1.to(device), y_batch_z1.to(device), z_batch_z1.to(device)
            
            Yhat = net(xz_batch)
            Yhat_z0, Yhat_z1 = Yhat[z_batch==0], Yhat[z_batch==1]
            m = z_batch.shape[0]

            # prediction loss for different subgroups
            loss_z0 = loss_function(Yhat_z0.squeeze(), y_batch_z0)
            loss_z1 = loss_function(Yhat_z1.squeeze(), y_batch_z1)
            l2_reg = reg_strength * ((p1 - q_z0)**2)
            total_loss = (1 - lambda_) * (p1/q_z0 * loss_z0 + (1-p1)/q_z1 * loss_z1) - l2_reg
            
            # DP_Constraint
            Pr_Ytilde1 = CDF_tau(Yhat.detach(),h,tau)
            for z in sensitive_attrs:
                Pr_Ytilde1_Z = CDF_tau(Yhat.detach()[z_batch==z],h,tau)
                m_z = z_batch[z_batch==z].shape[0]

                Delta_z = Pr_Ytilde1_Z-Pr_Ytilde1
                Delta_z_grad = torch.dot(phi((tau-Yhat.detach()[z_batch==z])/h).view(-1), 
                                            Yhat[z_batch==z].view(-1))/h/m_z
                Delta_z_grad -= torch.dot(phi((tau-Yhat.detach())/h).view(-1), 
                                            Yhat.view(-1))/h/m

                if Delta_z.abs() >= delta:
                    if Delta_z > 0:
                        Delta_z_grad *= lambda_*delta
                        total_loss += Delta_z_grad
                    else:
                        Delta_z_grad *= -lambda_*delta
                        total_loss += Delta_z_grad
                else:
                    Delta_z_grad *= lambda_*Delta_z
                    total_loss += Delta_z_grad
            
            # Minimum optimization for the model parameter
            optimizer_m.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer_m.step()
            costs.append(total_loss.item())
            
            # Maximum optimization for the robust regularization
            optimizer_p.zero_grad()
            neg_loss = -total_loss
            neg_loss.backward()
            optimizer_p.step()
            
            # Projection to the range (0, 1)
            with torch.no_grad():
                p1.clamp_(0, 1)
                
            print(p1)
            
            if (i + 1) % k == 0 or (i + 1) == len(data_loader):
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch+1, n_epochs,
                                                                          i+1, len(data_loader),
                                                                          total_loss.item()), end='\r')
                
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        Yhat_train = net(XZ_train).squeeze().detach().cpu().numpy()
        df_temp = measures_from_Yhat(Y_train_np, Z_train_np, Yhat=Yhat_train, threshold=tau)
        df_temp['epoch'] = epoch * len(data_loader) + i + 1
        df_ckpt = df_ckpt.append(df_temp)
    
    # Results for different subgroups based on sensitive attributes
    Yhat_test = net(XZ_test).squeeze().detach().cpu().numpy()
    df_test = measures_from_Yhat(Y_test_np, Z_test_np, Yhat=Yhat_test, threshold=tau)
    
    Yhat_test_z0 = net(XZ_test_z0).squeeze().detach().cpu().numpy()
    df_test_z0 = measures_from_Yhat(Y_test_np_z0, Z_test_np_z0, Yhat=Yhat_test_z0, threshold=tau)
    
    Yhat_test_z1 = net(XZ_test_z1).squeeze().detach().cpu().numpy()
    df_test_z1 = measures_from_Yhat(Y_test_np_z1, Z_test_np_z1, Yhat=Yhat_test_z1, threshold=tau)
    
    return df_test, df_test_z0, df_test_z1