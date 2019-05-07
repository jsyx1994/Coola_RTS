import torch
from torch import nn
import torch.nn.functional as F


class CriticHead(nn.Module):

    def __init__(self, in_features):
        super(CriticHead, self).__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

