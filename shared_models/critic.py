import torch
from torch import nn
import torch.nn.functional as F
from config import model_saved_dir


class CriticHead(nn.Module):
    def __init__(self, in_features):
        super(CriticHead, self).__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(1)

        try:
            self.load_state_dict(torch.load(model_saved_dir + '/critic_head.pt'))
        except FileNotFoundError as e:
            print(e, ', File will be touched')
            torch.save(self.state_dict(), model_saved_dir + '/critic_head.pt')

    def __del__(self):

        # pass
        torch.save(self.state_dict(), model_saved_dir + '/critic_head.pt')
        # print('critic head saved')

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

