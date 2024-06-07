import os
import copy
import torch
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device)

def arrays_to_tensor_1(X, Y, Z, XZ, Z_2, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device), torch.FloatTensor(Z_2).to(device)

def compas_data_loader():
    """ Downloads COMPAS data from the propublica GitHub repository.
    :return: pandas.DataFrame with columns 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'two_year_recid', 'age_cat_25 - 45',
       'age_cat_Greater than 45', 'age_cat_Less than 25', 'race_African-American',
       'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'
    """
    data = pd.read_csv("./data/compas/compas-scores-two-years.csv")
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]

    data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]

    data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]

    data = data.assign(sex=(data["sex"] == "Male") * 1)
    data = pd.get_dummies(data)
    return data


class CustomDataset():
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessDataset():
    def __init__(self, dataset, device=torch.device('cuda')):
        self.dataset = dataset
        self.device = device
        self.seed = 1234
        
        if self.dataset == 'Adult':
            self.get_adult_data()
        elif self.dataset == 'COMPAS':
            self.get_compas_data()
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
        self.prepare_ndarray()
    
    def get_adult_data(self):
        with open('./data/adult/adult.data') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            y = []
            z = []

            i = 0
            for row in csv_reader:
                if i == 0:
                    i += 1
                    continue
                
                if row[9] == 'Male':
                    z.append(1)
                else:
                    z.append(0)

                if row[14] == '>50K':
                    y.append(1)
                else:
                    y.append(0)

        data = pd.read_csv("./data/adult/AdultTrain.csv")
        data = pd.get_dummies(data)
        data.drop(['fnlwgt', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other'], axis=1)
        Y = pd.Series(y, name='>50K')
        Z = pd.Series(z, name='gender')
        
        data = pd.concat([data, Y, Z], axis=1)
        data = data.sample(frac=1, random_state=self.seed)

        df_train = data.iloc[0:25000]
        df_test = data.iloc[25000:32500]

        df_train_1 = df_train.loc[df_train[('gender')] == 1]
        df_train_2 = df_train.loc[df_train[('gender')] == 0]

        df_test_1 = df_test.loc[df_test[('gender')] == 1]
        df_test_2 = df_test.loc[df_test[('gender')] == 0]

        df_train_1 = df_train_1.iloc[0:12000]
        df_train_2 = df_train_2.iloc[0:3000]

        df_test_1 = df_test_1.iloc[0:4000]
        df_test_2 = df_test_2.iloc[0:1000]

        df_train = pd.concat([df_train_1, df_train_2])

        df_test = pd.concat([df_test_1, df_test_2])

        self.Z_train_ = df_train[('gender')].values
        self.Z_test_ = df_test[('gender')].values

        self.Y_train_ = df_train[('>50K')].values
        self.Y_test_ = df_test[('>50K')].values

        self.X_train_ = df_train.drop(['gender', '>50K'], axis=1).values
        self.X_test_ = df_test.drop(['gender', '>50K'], axis=1).values
        
    def get_compas_data(self):
        dataset = compas_data_loader()
        data_new = dataset.drop(['race_African-American'], axis=1)
        
        data_new = data_new.sample(frac=1, random_state=self.seed)
        
        df_train = data_new.iloc[0:4000]
        df_test = data_new.iloc[4000:5000]
        
        df_train_1 = df_train.loc[df_train[('race_Caucasian')] == 1]
        df_train_2 = df_train.loc[df_train[('race_Caucasian')] == 0]
        
        df_test_1 = df_test.loc[df_test[('race_Caucasian')] == 1]
        df_test_2 = df_test.loc[df_test[('race_Caucasian')] == 0]
        
        df_train_1 = df_train_1.iloc[0:500]
        df_train_2 = df_train_2.iloc[0:2000]
        
        df_test_1 = df_test_1.iloc[0:150]
        df_test_2 = df_test_2.iloc[0:600]
        
        df_train = pd.concat([df_train_1, df_train_2])
        
        df_test = pd.concat([df_test_1, df_test_2])
        
        self.Z_train_ = df_train[('race_Caucasian')].values
        self.Z_test_ = df_test[('race_Caucasian')].values
        
        self.Y_train_ = df_train[('two_year_recid')].values
        self.Y_test_ = df_test[('two_year_recid')].values
        
        self.X_train_ = df_train.drop(['race_Caucasian','two_year_recid'], axis=1).values
        self.X_test_ = df_test.drop(['race_Caucasian','two_year_recid'], axis=1).values


    def prepare_ndarray(self):
        self.normalized = False
        self.X_train = self.X_train_
        
        self.Y_train = self.Y_train_
        
        self.Z_train = self.Z_train_
        
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_
        
        self.Y_test = self.Y_test_
        
        self.Z_test = self.Z_test_
        
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None
        
    def normalize(self):
        self.normalized = True
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)
        
        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_test = scaler_X.transform(self.X_test)
        
        return None
    
    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test),\
    
    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_= arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_= arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)

        
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)
    