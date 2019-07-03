import socket
import json
from game.rts import RtsUtils
import multiprocessing as mp
from multiprocessing import Process
from ai.bot import Bot
import torch
from shared_models.model import ActorCritic, TestLeakModel
from torch.distributions import Categorical
import numpy as np
from game.self_defined_actions import WorkerAction
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.multiprocessing as tmp
from collections import namedtuple
from game.env import Environment

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'location'))

env = Environment()
# rts_utils = RtsUtils()
# worker = ActorCritic(actor='Worker')  # output  critic or worker policy
testmodel = TestLeakModel()


class ServerAI:
    def __init__(self, host='127.0.0.1', port=9898):
        self.host = host
        self.port = port
        self.client_num = 0

    def run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
            ss.bind((self.host, self.port))
            print('Waiting for a client connection...')
            while 1:
                ss.listen()
                conn, addr = ss.accept()
                # print(type(conn))
                self.client_num += 1
                # print('incoming client {} conne')
                SocketWrapperAI(conn, self.client_num).run_episodes()
                # mp.Process(target=SocketWrapperAI(conn, self.client_num).run_episodes, args=()).start().


class SocketWrapperAI:
    DEBUG = 0
    GAMMA = 0.99

    def __init__(self, client_socket: socket.socket, client_number):
        self.client_socket = client_socket
        self.client_number = client_number
        self.client_socket.settimeout(1000)

        if self.DEBUG >= 1:
            print("New connection with client# {} at {}".format(client_number, client_socket))

        self.actor_map = {'Worker': ActorCritic(actor='Worker')}
        self.worker_memory = []
        self.state = None
        self.G0 = 0
        self.G0s = []

    def init_new_episode(self):
        if self.DEBUG == 3:
            import gc
            gc.collect()
            import objgraph
            objgraph.show_growth()
        global worker
        self.actor_map['Worker'] = ActorCritic(actor='Worker')

        self.worker_memory = []
        self.state = None
        self.G0 = 0

    @staticmethod
    def select_action(unit_nn, state, info):
        with torch.no_grad():
            # print(unit_nn(state,info))
            return int(Categorical(unit_nn(state, info)).sample()[0])  # a(t)

    def sample(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float()
        assignable = env.rts_utils.get_assignable()
        act_loc = {'Worker': [], 'Light': [], 'Heavy': [], 'Base': [], 'Barracks': []}
        # actions = {'Worker': [], 'Light': [], 'Heavy': [], 'Base': [], 'Barracks': []}
        # locations = {'Worker': []}
        for unit in assignable:
            location = int(unit['x']), int(unit['y'])
            actor = unit['type']
            loc = torch.LongTensor(location)
            if actor == 'Worker':   # if is for test
                action = self.select_action(self.actor_map[actor], state, info=loc)
                env.rts_utils.translate_action(uid=unit['ID'], location=location, bot_type=actor, act_code=action)
                act_loc[actor].append((action, loc))
        return env.rts_utils.get_player_action(), act_loc

    @staticmethod
    def record(memory: list, state, act_loc, next_state, reward):
        for action, location in act_loc:
            memory.append(
                Transition(state, action, next_state, reward, location))

    def run_episodes(self):
        """
        If you don't understand the main logic behind santi's work, 
        please carefully go through the comments of this function.
        """
        client_socket = self.client_socket

        welcome = 'PyJSONSocketWrapperAI: you are client #%i\n' % self.client_number
        client_socket.sendall(welcome.encode())

        episodes = 0
        first_step = True
        while 1:
            if first_step:
                self.init_new_episode()
                first_step = False
                done = False
                episodes += 1
                msg = str(client_socket.recv(10240).decode())
                if msg == 0:
                    print('break')
                    break
                client_socket.sendall(('ack\n').encode())
                gs = msg.split('\n')[1]
                gs = json.loads(gs)
                player = int(msg.split()[1])
                self.state, _, _, = env.reset(gs, player)
            else:
                while (not done):
                    state = self.state
                    pa, act_loc = self.sample(state)
                    next_state, reward, done = env.step(client_socket, pa, env.rts_utils.get_player())
                    self.state = next_state
                    for act, _ in act_loc['Worker']:
                        if act != WorkerAction.NO_OP:
                            self.G0 += 1
                    self.G0 += reward
                    self.record(memory=self.worker_memory, state=state, act_loc=act_loc['Worker'],
                                next_state=next_state, reward=reward)

                self.G0s.append(self.G0)
                self.plot_durations()
                self.optimize(actor=self.actor_map['Worker'], batch_bundle=self.worker_memory)
                first_step = True
        client_socket.close()

    def optimize(self, actor, batch_bundle):
        batch_bundle = [ts for ts in batch_bundle if ts.action]
        if len(batch_bundle) == 0:
            print('skip training')
            return
        batch = Transition(*zip(*batch_bundle))

        # return
        # print('s', len(batch.state))
        # print('s\'', len(batch.next_state))
        # print('act', len(batch.action))
        # print('reward', len(batch.reward))
        # print('loc', len(batch.location))
        optimizer = optim.Adam(params=actor.parameters(), lr=1e-3)

        state_batch = torch.Tensor(batch.state)
        next_state_batch = torch.Tensor(batch.next_state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.Tensor(batch.reward)
        # print(batch.location)
        location_batch = torch.cat(batch.location)

        # print(location_batch)
        v_t = actor(input=state_batch, info='critic')
        # print('v_t',v_t.size())
        v_t_1 = actor(input=next_state_batch, info='critic')
        # print('v_t_1',v_t_1.size())

        # pi = worker.forward(input=state_batch, info=location_batch)
        # pi_sa = worker.forward(input=state_batch, info=location_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        # print('pi(a|s)', pi_sa)
        policy = actor(input=state_batch, info=location_batch)
        log_pi_sa = torch.log(
            policy.gather(1, action_batch.unsqueeze(1)).squeeze())
        # print('log_pi_sa', log_pi_sa.size())
        targets = reward_batch.unsqueeze(1) + self.GAMMA * v_t_1
        # print('targets', targets.size())
        predicts = v_t
        td_error = targets - predicts
        # entropy = -torch.sum(policy[0] * torch.log(policy[0]))

        pg_loss = log_pi_sa * td_error
        ac_loss = -pg_loss.mean() + F.mse_loss(targets, predicts)   # - entropy
        # print(ac_loss.size())

        # print('loss', pg_loss.mean().size(), F.mse_loss(targets, predicts).size())
        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()

        # print(state_batch.size())
        # print(state_batch.size())
        # print(next_state_batch.size())
        # print(action_batch.size())
        # print(reward_batch.size())

    def plot_durations(self):
        # print(self.G0)
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.G0s, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        # plt.ylabel('Duration')
        plt.ylabel('Returns')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

def test():
    x = torch.randn((1, 18, 8, 8))
    # import gc
    while 1:
        # testmodel = TestLeakModel()
        print('loop one')
        # gc.collect()
        for i in range(10000):
            with torch.no_grad():
                forward(testmodel, x)
    # worker = ActorCritic(actor='Worker')


def forward(model, x):
    return np.asarray(model.forward(x))


def test1():
    while (1):
        worker = ActorCritic(actor='Worker')
        for i in range(3000):
            worker.forward(torch.rand(1, 18, 8, 8))


if __name__ == "__main__":
    # test()
    # test1()
    # import time
    #
    # time.sleep(10)
    # babyAI = BabyAI()
    # mp.set_start_method('spawn')
    # print(mp.get_start_method())
    ServerAI().run_server()
