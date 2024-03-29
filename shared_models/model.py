import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.convlstm import ConvLSTM
from utils.self_attention import MultiHeadedAttention
from shared_models.actor import ActorHead
from shared_models.critic import CriticHead
from config import model_saved_dir
from game.setttings import map_size

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Pic2Vector(nn.Module):
    """
    convent matrix to Vectors
    each pixel is like embedding to it's channel's dimension
    """
    def forward(self, input):
        batch_size, channel_size = input.size(0), input.size(1)
        return input.view(batch_size, channel_size, -1).transpose(-2, -1)  # transpose to keep d_model last


class Shared(nn.Module):
    """
    this is a common model for all agents
    """
    def __init__(self, minimap_size=map_size, input_channel=18):
        """
        The internal_state indicate the i_th hidden layers's output and cell state
        E.G. internal_state[0][0](torch.Size([1, 16, 6, 6])) indicate the layer 0's hidden output
        and [0][1] indicate the layer's cell state(torch.Size([1, 16, 6, 6])).

        We need to have access to the internal_state through one episode in one RTS battle,
        store it and then pass them to the next forwarding.
        """
        super(Shared, self).__init__()
        self.minimap_size = minimap_size
        self.input_channel = input_channel
        self.last_lstm_dim = 32
        self.last_shared_layer_dim = minimap_size[0] * minimap_size[1] * self.last_lstm_dim    # calc the output dim of the net

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, padding=1)   # WEIGHTs ARE INITED
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4_bn = nn.BatchNorm2d(32)

        # self.convLSTM = ConvLSTM(input_channels=32, hidden_channels=[32, self.last_lstm_dim], kernel_size=3)
        # self.convLSTM_bn = nn.BatchNorm2d(32)

        self.wv = Pic2Vector()
        self.self_attention = MultiHeadedAttention(h=4, d_model=32)

        self.flatten = Flatten()
        try:
            self.load_state_dict(torch.load(model_saved_dir + '/shared.pt'))
        except FileNotFoundError as e:
            print(e, ', Model file will be touched')
            torch.save(self.state_dict(), model_saved_dir + '/shared.pt')
        self.internal_state = []
        print(self.internal_state)

    def forward(self, input,step):
        x = input
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.conv4_bn(F.relu(self.conv4(x)))
        # x = F.relu(self.conv5(x))

        # x, self.internal_state = self.convLSTM(x, step, self.internal_state)    # will change the initial state
        # x = self.convLSTM_bn(F.relu(x))
        # print(x.size())
        x = self.wv(x)
        # print("wv size:", x.size())
        x = F.relu(self.self_attention(x, x, x))   # apply relu activation for the returning Linear layer
        # print(x.size())
        # print(internal_state[1][1].size())
        x = self.flatten(x)
        # print(x.size())
        # print(len(internal_state))
        # print(x)
        return x
        # pass

    def __del__(self):
        """
        save shared paras
        :return:
        """
        # pass
        torch.save(self.state_dict(), model_saved_dir + '/shared.pt')
        # print('shared saved')


class ActorCritic(nn.Module):
    """
    load shared paras from common model and outputs

    Critic net = Shared_\theta + Critic head
    Actor net = Shared_\theta + Actor head
    """
    def __init__(self, actor):
        super(ActorCritic, self).__init__()
        self.actor = actor

        self.base = Shared()    # loads saved shared weights
        self.in_features = self.base.last_shared_layer_dim
        self.pi_out = ActorHead(self.in_features, actor)   # loads saved actor head weights
        self.v_out = CriticHead(self.in_features)  # loads saved critic head weights

    def forward(self, input, info='critic', step=0):
        x = self.base(input, step)
        # print(x.size())
        if isinstance(info, torch.Tensor):
            return self.pi_out.forward(x, loc=info)
        else:
            return self.v_out.forward(x)
        # if info == 'critic':
        #     return self.v_out.forward(x)    # only use critic
        # else:
        #     return self.pi_out.forward(x, loc=info)   # use both

    def __del__(self):
        print('deleted')
        # pass

class TestLeakModel(nn.Module):
    def __init__(self):
        super(TestLeakModel,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3, padding=1)   # WEIGHTs ARE INITED
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                               padding=1)  # WEIGHTs ARE INITED
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)  # WEIGHTs ARE INITED
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                               padding=1)  # WEIGHTs ARE INITED
        self.fc1 = nn.Linear(in_features=64*16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=7)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = F.softmax(x,dim=1)
        return x

def test():
    net = ActorCritic()
    torch.save(net.state_dict(), model_saved_dir + '/worker.pt')
    print(net)
    # params = list(net.parameters())
    # print(len(params))
    # for x in params:
    #     print(x.size())

    # print(list(net.parameters()[0]))
    import time
    # print(type(x))

    start = time.time()
    pi, v = net(torch.rand(1, 20, 8, 8))
    print(pi, '\n', v)
    print(time.time() - start)
    # print(x)
    # print(x)


def test1():
    # different weight initial
    model = Shared()
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor][0][0])
        break

    model = Shared()
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor][0][0])
        break


def test2():
    """
    Shared para load/save/write test
    :return:
    """
    net = ActorCritic(actor='Worker')
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor][0][0])
        net.state_dict()[param_tensor][0][0][0][0] *= -1
        break


def test3():
    from ai.bot import Bot
    from game.rts import RtsUtils
    gs = {'time': 1600, 'pgs': {'width': 32, 'height': 32,
                                'terrain': '0000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000011111000000001111100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000',
                                'players': [{'ID': 0, 'resources': 5}, {'ID': 1, 'resources': 20}], 'units': [
            {'type': 'Resource', 'ID': 8, 'player': -1, 'x': 30, 'y': 0, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 9, 'player': -1, 'x': 31, 'y': 0, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 10, 'player': -1, 'x': 30, 'y': 1, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 11, 'player': -1, 'x': 31, 'y': 1, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 12, 'player': -1, 'x': 0, 'y': 30, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 13, 'player': -1, 'x': 1, 'y': 30, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 14, 'player': -1, 'x': 0, 'y': 31, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 15, 'player': -1, 'x': 1, 'y': 31, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 20, 'player': -1, 'x': 2, 'y': 31, 'resources': 10, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 21, 'player': -1, 'x': 0, 'y': 29, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 22, 'player': -1, 'x': 29, 'y': 0, 'resources': 20, 'hitpoints': 1},
            {'type': 'Resource', 'ID': 23, 'player': -1, 'x': 31, 'y': 2, 'resources': 20, 'hitpoints': 1},
            {'type': 'Base', 'ID': 25, 'player': 0, 'x': 6, 'y': 14, 'resources': 0, 'hitpoints': 10},
            {'type': 'Worker', 'ID': 26, 'player': 0, 'x': 3, 'y': 10, 'resources': 0, 'hitpoints': 1},
            {'type': 'Base', 'ID': 27, 'player': 1, 'x': 25, 'y': 17, 'resources': 0, 'hitpoints': 10},
            {'type': 'Worker', 'ID': 28, 'player': 1, 'x': 25, 'y': 19, 'resources': 0, 'hitpoints': 1},
            {'type': 'Barracks', 'ID': 29, 'player': 0, 'x': 4, 'y': 10, 'resources': 0, 'hitpoints': 4},
            {'type': 'Base', 'ID': 30, 'player': 0, 'x': 2, 'y': 10, 'resources': 0, 'hitpoints': 10}]}, 'actions': []}
    # print(Bot(bot_type='Worker').predict(torch.randn(1, 20, 8, 8)))
    rts_utils = RtsUtils()
    rts_utils.reset(gs, 0)
    assignable = rts_utils.get_assignable()
    x = rts_utils.parse_game_state()

    x = torch.from_numpy(x).float().unsqueeze(0)

    bot = Bot(bot_type='Worker')
    for unit in assignable:
        if unit['type'] == 'Worker':
            rts_utils.translate_action(uid=unit['ID'], location=(int(unit['x']), int(unit['y'])), bot_type='Worker',
                                       act_code=bot.decide(x))
    print(rts_utils.get_player_action())


def test4():
    """
    value head load test

    Expect: the [0][0] changes each time from the model
    :return:
    """
    net = ActorCritic(actor='Worker')
    print(net)
    print('v_out.fc1.weight00', "\t", net.state_dict()['v_out.fc1.weight'][0][0])
    net.state_dict()['v_out.fc1.weight'][0][0] *= -1
    pass

def test5():
    loc = torch.tensor((7, 7))
    s = torch.zeros(1,18,8,8)
    net = ActorCritic(actor='Worker')

    print(net.forward(s, info=loc))
    loc = torch.tensor((6, 7))
    print(net.forward(s, info=loc))

def test_sensitive():
    net = TestLeakModel()
    s = torch.rand(1,18,8,8)
    x1 = torch.zeros(1,1,8,8)
    x1[0][0][1][1] = 1
    x2 = torch.zeros(1, 1, 8, 8)
    x2[0][0][0][7] = 1
    # print(net(torch.cat((s,x1),dim=1)))
    # print(net(torch.cat((s,x2),dim=1)))
    #
    x1.fill_(7/7)
    y1 = torch.zeros(x1.shape)
    y1.fill_(1/7)
    print(net(torch.cat((s, x1, y1), dim=1)))

    x1.fill_(1/7)
    y1.fill_(2/7)

    print(net(torch.cat((s,x1,y1),dim=1)))


if __name__ == '__main__':
    test_sensitive()
    pass
