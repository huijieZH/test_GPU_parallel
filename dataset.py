from torch.utils.data import Dataset
import torch
import time

class SampleDataset(Dataset):
    def __init__(self, data_size = 32):
        self.data_size = data_size

    def __getitem__(self, idx):
        datadict = {}
        datadict['rgb'] = torch.randn(4, self.data_size, self.data_size)
        datadict['time_step'] = torch.randn(1)
        return datadict

    def __len__(self):
        return 10000