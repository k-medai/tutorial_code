import wfdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_X(df, sampling_rate, signal_path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(signal_path+f) for f in df.filename_lr]
    elif sampling_rate == 500:
        data = [wfdb.rdsamp(signal_path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def load_all(csv_path, split, signal_path, sampling_rate= 500):
    data = pd.read_csv(f'{csv_path}ptbxl_super_class_{split}.csv')
    X = load_X(data, sampling_rate= sampling_rate, signal_path= signal_path)
    y = np.array([data.iloc[i,-5:].values.tolist() for i in range(len(data))])
    return X,y

class Dataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype= torch.float32).permute(0,2,1)
        self.y = torch.tensor(y, dtype= torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

def make_dataloader(dataset, batch_size= 16, shuffle= True):
    return DataLoader(dataset, batch_size, shuffle= shuffle)

def load_dataloader(csv_path, signal_path, sampling_rate= 500, batch_size= 16, shuffle= False):
    # train
    X_train, y_train = load_all(csv_path, 'train', signal_path, sampling_rate= sampling_rate)
    trainloader = make_dataloader(Dataset(X_train, y_train), batch_size= batch_size, shuffle= shuffle)
    # valid
    X_valid, y_valid = load_all(csv_path, 'val', signal_path, sampling_rate= sampling_rate)
    validloader = make_dataloader(Dataset(X_valid, y_valid), batch_size= batch_size, shuffle= shuffle)
    # test
    X_test, y_test = load_all(csv_path, 'test', signal_path, sampling_rate= sampling_rate)
    testloader = make_dataloader(Dataset(X_test, y_test), batch_size= batch_size, shuffle= shuffle)
    return trainloader, validloader, testloader