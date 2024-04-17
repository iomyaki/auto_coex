import torch.nn as nn
from torch.utils.data import Dataset


class AE(nn.Module):
    def __init__(self, input_size, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, input_size), nn.ReLU())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return int(self.data.size(dim=0))

    def __getitem__(self, index):
        return {"noise_data": self.data[index][0], "orig_data": self.data[index][1]}
