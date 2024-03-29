import torch
from torch import nn
import torch.nn.functional as F
from game.self_defined_actions import *
from game.setttings import units_type, map_size
from config import model_saved_dir
import functools


class ActorHead(nn.Module):
    
    def __init__(self, in_features, model_name):
        super(ActorHead, self).__init__()
        assert model_name in units_type
        self.model_name = model_name

        self.fc1 = nn.Linear(in_features=in_features + map_size[0] * map_size[1], out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.logits = functools.partial(nn.Linear, in_features=64)

        if model_name == 'Worker':
            self.logits = self.logits(out_features=WorkerAction.NUMBER_OF_ACTIONS)

        elif model_name == 'Base':
            pass
        elif model_name == 'Barracks':
            pass
        elif model_name == 'Light':
            pass
        elif model_name == 'Heavy':
            pass
        elif model_name == 'Ranged':
            pass

        try:
            self.load_state_dict(torch.load(model_saved_dir + '/' + model_name + '_head.pt'))
        except FileNotFoundError as e:
            print(e, ' File will be touched')
            torch.save(self.state_dict(), model_saved_dir + '/' + self.model_name + '_head.pt')
        # self.logits = nn.Linear(256, self.action_dim)

    def forward(self, input, loc):
        """
        :param input: global state
        :param loc: unit location
        :return: policy
        """
        # assert input.size(0) == loc.size(0)

        x = input
        batch_size = x.size(0)
        x = x.view((batch_size, -1, map_size[0], map_size[1]))

        channel_location = torch.zeros((batch_size, 1, map_size[0], map_size[1]))
        # x_f = torch.zeros((batch_size, 1, map_size[0], map_size[1]))
        # y_f = torch.zeros((batch_size, 1, map_size[0], map_size[1]))

        # torch.scatter()
        for b in range(batch_size):
            x_, y_,  = loc[b]
            # print(loc[b])
            channel_location[b][0][x_][y_] = 1
            # x_f[b][0].fill_(x_)
            # y_f[b][0].fill_(y_)
        # print(channel_location)
            # print(channel_location[b])
        # channel_location[0][loc[0]][loc[1]] = 1
        # print(channel_location.size())
        # print(x.size())
        x = torch.cat((x, channel_location), dim=1)
        # print(x[0][-1])
        # print(x[0][-2])
        # print(x.size())
        x = x.view(batch_size, -1)

        # print(x.size())
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3_bn(F.relu(self.fc3(x)))

        # x = self.fc2(x)
        # if x.size(0) != 1:
        #     x = F.relu(self.fc7_bn(x))
        #     print('linear batch normed')
        # else:
        #     x = F.relu(x)
        x = F.softmax(self.logits(x), dim=-1)
        return x

    def __del__(self):
        # pass
        torch.save(self.state_dict(), model_saved_dir + '/' + self.model_name + '_head.pt')
        # print(self.model_name + '_head.pt saved')


# def test():

if __name__ == '__main__':
    model = ActorHead(in_features=128, model_name='Worker')
    print(model)