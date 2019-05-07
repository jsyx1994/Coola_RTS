import torch
from torch import nn
import torch.nn.functional as F
from utils.self_defined_actions import *

class ActorHead(nn.Module):
    
    def __init__(self, in_features):
        super(ActorHead, self).__init__()
        self.action_dim = WorkerAction.NUMBER_OF_ACTIONS

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.logits = nn.Linear(256, self.action_dim)

    def forward(self, input):
        x = input
        x = F.relu(self.fc1(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x





# print(actor_m(shared))

def model_save_test():
    torch.save(model.state_dict(), './model.pt')
    model.load_state_dict(torch.load('./model.pt'))
    print('internal_state = ', model.initial_state)