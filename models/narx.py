import torch
import torch.nn as nn
import torch.nn.functional as F

class Scaler(nn.Module):
    def __init__(self, rate): 
        super().__init__()
        self.rate = rate
    def forward(self, x):
        return x / self.rate if self.training else x


class WaterNARX(nn.Module):
    def __init__(self, in_dim, hidden_dim, rate):
        super().__init__()
        hid = int(hidden_dim * rate)

        self.proj = nn.Linear(in_dim, hid)
        self.h1 = nn.Linear(hid, hid)
        self.bn1 = nn.BatchNorm1d(hid, track_running_stats=False)
        self.h2 = nn.Linear(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid, track_running_stats=False)

        self.head = nn.Linear(hid, 1)
        self.scaler = Scaler(rate)

    def forward(self, water, rain):
        b = water.size(0)
        w = water.view(b, -1)
        r = rain.view(b, -1)

        x = F.relu(self.proj(torch.cat((w, r), dim=1)))
        x = self.scaler(x)
        x = self.scaler(F.relu(self.bn1(self.h1(x))))
        x = self.scaler(F.relu(self.bn2(self.h2(x))))
        return self.head(x)
