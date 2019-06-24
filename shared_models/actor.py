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
        self.logits = functools.partial(nn.Linear, in_features=256)

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
        print(batch_size)
        x = x.view((-1, map_size[0], map_size[1]))
        # print(x.size())

        channel_location = torch.zeros((1, map_size[0], map_size[1]))
        # torch.scatter()
        channel_location[0][loc[0]][loc[1]] = 1
        x = torch.cat((x, channel_location))
        x = x.flatten().unsqueeze(0)

        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x

    def __del__(self):
        # pass
        torch.save(self.state_dict(), model_saved_dir + '/' + self.model_name + '_head.pt')
        print(self.model_name + '_head.pt saved')


# def test():



if __name__ == '__main__':
    model = ActorHead(in_features=128, model_name='Worker')
    print(model)