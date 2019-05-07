import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.convlstm import ConvLSTM
from utils.self_attention import MultiHeadedAttention
from models.actor import ActorHead
from models.critic import CriticHead


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
    def __init__(self, minimap_size=(8, 8), input_channel=20):
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
        self.initial_state = []
        self.last_lstm_dim = 32
        self.last_shared_layer_dim = minimap_size[0] * minimap_size[1] * self.last_lstm_dim    # calc the output dim of the net

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1)   # WEIGHTs ARE INITED
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.convLSTM = ConvLSTM(input_channels=32, hidden_channels=[16, self.last_lstm_dim], kernel_size=3)
        self.wv = Pic2Vector()
        self.self_attention = MultiHeadedAttention(h=4, d_model=32)
        self.flatten = Flatten()

    def forward(self, input):
        x = input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, internal_state = self.convLSTM(x, self.initial_state)
        # print(x.size())
        x = self.wv(x)
        # print("wv size:", x.size())
        x = F.relu(self.self_attention(x, x, x))    # apply relu activation for the returning Linear layer
        # print(x.size())
        # print(internal_state[1][1].size())
        x = self.flatten(x)
        # print(x.size())
        # print(len(internal_state))
        # print(x)
        return x
        # pass


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.base = Shared()

        self.in_features = self.base.last_shared_layer_dim
        self.pi_out = ActorHead(self.in_features)
        self.v_out = CriticHead(self.in_features)

    def forward(self, input):
        x = self.base(input)
        return self.pi_out(x), self.v_out(x)


if __name__ == '__main__':


    net = ActorCritic()
    torch.save(net.state_dict(), './worker.pt')
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
    print(time.time()-start)
    # print(x)
    # print(x)