from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd


class HouseDataset(Dataset):
    def __init__(self, mode='train', fold=0):
        if mode == 'train':
            self.df = pd.read_csv(f'./data/train_norm{fold}.csv')
        elif mode == 'val':
            self.df = pd.read_csv(f'./data/val_norm{fold}.csv')
        elif mode == 'test':
            self.df = pd.read_csv('./data/test_norm.csv')
        elif mode == 'all_data':
            self.df = pd.read_csv('./data/all_data.csv')
        else:
            raise NotImplementedError('No such mode')
        self.features = [i for i in list(self.df.columns) if i != 'price']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row['price']
        features = row[self.features]

        return torch.Tensor(features), target


def data_loader(batch_size=8, fold=0):
    data_train = HouseDataset(mode='train', fold=fold)
    data_val = HouseDataset(mode='val', fold=fold)
    data_test = HouseDataset(mode='test')
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return {'train': train_loader, 'val':val_loader, 'test': test_loader}
