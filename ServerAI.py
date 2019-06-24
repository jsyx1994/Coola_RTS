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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'location'))

rts_utils = RtsUtils()
# worker = ActorCritic(actor='Worker')    # output  critic or worker policy
testmodel = TestLeakModel()


# worker = ActorCritic(actor='Worker')


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
                self.client_num += 1
                # print('incoming client {} conne')
                SocketWrapperAI(conn, self.client_num).run_episodes()
                # mp.Process(target=SocketWrapperAI(conn, self.client_num).run_episodes, args=()).start().


class SocketWrapperAI:
    DEBUG = 0
    GAMMA = 0.99

    def __init__(self, client_socket, client_number, ai=None):
        self.client_socket = client_socket
        self.client_number = client_number
        # utt: UnityTypeTable
        self.utt = None
        self.ai = ai
        if self.DEBUG >= 1:
            print("New connection with client# {} at {}".format(client_number, client_socket))

        self.worker_memory = []
        self.p_state = None
        self.p_worker_info = {}  # a dict store all worker's actions {id->(action,location)}
        self.G0 = 0

    def step(self, action):
        pass

    def init_new_episode(self):
        if self.DEBUG == 3:
            import gc
            gc.collect()
            import objgraph
            objgraph.show_growth()

        global worker
        worker = ActorCritic(actor='Worker')

        self.worker_memory = []
        self.p_worker_info = {}  # a dict store all worker's actions {id->(action,location)}
        self.p_state = None
        self.G0 = 0

    def run_episodes(self):
        """
        If you don't understand the main logic behind santi's work, 
        please carefully go through the comments of this function.
        """
        with self.client_socket as client_socket:
            welcome = 'PyJSONSocketWrapperAI: you are client #%i\n' % self.client_number
            client_socket.sendall(welcome.encode())
            while 1:
                # retrieve the message from socket
                msg = str(client_socket.recv(10240).decode())
                # print(msg)
                # connection broken, retry.
                if msg == 0:
                    print('break')
                    break
                # client closes the connection.
                elif msg.startswith('end'):
                    if self.DEBUG >= 1:
                        print('end')
                    raise NotImplementedError
                    # self.client_socket.close()
                    # exit(self)

                # budget initialization
                elif msg.startswith('budget'):
                    """
                    reset the environment
                    """
                    if self.DEBUG >= 1:
                        print('game reset')

                    self.init_new_episode()
                    # msg = msg.split()
                    # self.ai.reset()
                    # timeBudget = int(msg[1])
                    # iterationsBudget = int(msg[2])
                    # if self.DEBUG >= 1:
                    #     print("setting the budget to: {},{}".format(self.ai.timeBudget, self.ai.iterationsBudget))
                    # send back ack
                    client_socket.sendall('ack\n'.encode())

                    # utt initialization
                elif msg.startswith('utt'):
                    if self.DEBUG >= 1:
                        print('utt')

                    msg = msg.split('\n')[1]

                    # json.loads() returns dict
                    self.utt = json.loads(msg)

                    if self.DEBUG >= 2:
                        print('setting the utt to: {}'.format(self.utt))
                        pass

                    # send back ack
                    client_socket.sendall('ack\n'.encode())

                    # client asks server to return units actions for current game state
                elif msg.startswith('getAction'):

                    # get player ID
                    player = int(msg.split()[1])  ### issue
                    # get game state
                    gs = msg.split('\n')[1]
                    # json.loads() returns dict
                    gs = json.loads(gs)
                    rts_utils.reset(gs, player)
                    assignable = rts_utils.get_assignable()
                    # busy = rts_utils.get_self_busy()    # id
                    # units = rts_utils.get_self_units()  # unit
                    # print(busy)

                    state = torch.from_numpy(rts_utils.parse_game_state()).unsqueeze(0).float()  # s(t)
                    p_r = rts_utils.get_last_reward()
                    self.G0 += p_r
                    if p_r > 0:
                        x = 2
                        pass
                    # print(p_r)
                    # for i in range(len(self.p_worker_action)):
                    #     a = p_r
                    #     pass
                    # print(Transition(0, 0, 0, p_r))
                    # print(len(self.p_worker_action))
                    for i in self.p_worker_info.values():
                        self.worker_memory.append(Transition(self.p_state, i[0], state, p_r, i[1]))
                        if p_r == 4:
                            print(self.worker_memory[-1])

                    # print(self.p_state)

                    # if gs['time'] % 1 == 0:
                    for unit in assignable:
                        location = int(unit['x']), int(unit['y'])
                        if unit['type'] == 'Worker':
                            # print(worker.forward(state, loc=location))
                            # if rts_utils.is_assignable(unit):
                            loc = torch.LongTensor(location)
                            action = self.select_action(worker, state, info=loc)

                            rts_utils.translate_action(uid=unit['ID'], location=location,
                                                       bot_type='Worker', act_code=action)
                            # print(type(unit['ID']))
                            self.p_worker_info[str(unit['ID'])] = (action, loc)

                    self.p_state = state  # assign to previous states

                    pa = rts_utils.get_player_action()
                    if self.DEBUG >= 2:
                        print('getAction for player {}'.format(player))
                        print('with game state %s' % gs)

                    client_socket.sendall(('%s\n' % pa).encode())

                    if self.DEBUG >= 1:
                        print('action sent!')

                # get preGameAnalysis
                elif msg.startswith('preGameAnalysis'):
                    if self.DEBUG >= 1:
                        print('PreGameAnalysis')
                    raise NotImplementedError

                elif msg.startswith('gameOver'):
                    msg = msg.split()
                    winner = msg[1]
                    if self.DEBUG >= 1:
                        print('gameOver, winner is %s' % winner)
                    # self.ai.game_over(winner)

                    client_socket.sendall(('ack\n').encode())
                    # tmp.set_start_method('fork')
                    # tmp.Process(target=self.optimize, args=()).start()

                    # add last
                    p_r = rts_utils.get_last_reward()
                    self.G0 += p_r

                    self.plot_durations()
                    for i in self.p_worker_info.values():
                        self.worker_memory.append(Transition(self.p_state, i[0], state, p_r, i[1]))
                    self.optimize()
                    # self.client_socket.close()

    @staticmethod
    def select_action(unit_nn, state, info):
        with torch.no_grad():
            return int(Categorical(unit_nn(state, info)).sample()[0])  # a(t)

    def optimize(self):
        # worker = ActorCritic(actor='Worker')
        global worker
        # print(self.worker_memory)
        print(worker)
        optimizer = optim.Adam(params=worker.parameters(), lr=1e-3)
        batch = Transition(*zip(*self.worker_memory))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.Tensor(batch.reward)
        # print(batch.location)
        location_batch = torch.cat(batch.location)
        # print(location_batch)
        v_t = worker.forward(input=state_batch, info='critic')
        print('v_t',v_t.size())
        v_t_1 = worker.forward(input=next_state_batch, info='critic')
        print('v_t_1',v_t_1.size())

        # pi = worker.forward(input=state_batch, info=location_batch)
        # pi_sa = worker.forward(input=state_batch, info=location_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        # print('pi(a|s)', pi_sa)
        log_pi_sa = torch.log(
            worker.forward(input=state_batch, info=location_batch).gather(1, action_batch.unsqueeze(1)).squeeze())
        print('log_pi_sa', log_pi_sa.size())
        targets = reward_batch.unsqueeze(1) + self.GAMMA * v_t_1
        print('targets', targets.size())
        predicts = v_t
        td_error = targets - predicts

        pg_loss = log_pi_sa * td_error
        ac_loss = -pg_loss.mean() +  F.mse_loss(targets, predicts)
        print(ac_loss.size())

        print('loss', pg_loss.mean().size(), F.mse_loss(targets, predicts).size())
        optimizer.zero_grad()
        ac_loss.backward(retain_graph=True)
        optimizer.step()


        print(state_batch.size())
        print(state_batch.size())
        print(next_state_batch.size())
        print(action_batch.size())
        print(reward_batch.size())


    def plot_durations(self):
        print(self.G_0s)
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.G_0s, dtype=torch.float)
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

        # Data for plotting
        # t = np.arange(0.0, 2.0, 0.01)
        # s = 1 + np.sin(2 * np.pi * t)
        #
        # fig, ax = plt.subplots()
        # ax.plot(t, s)
        #
        # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
        #        title='About as simple as it gets, folks')
        # ax.grid()
        #
        # fig.savefig("test.png")
        # plt.show()


def test():
    # for i in range(1000):
    #     worker = ActorCritic(actor='Worker')
    #     worker.forward(torch.rand(1, 18, 8, 8))
    # testmodel.forward(torch.rand(1, 18, 8, 8))
    # time.sleep(10)
    x = torch.randn((1, 18, 8, 8))
    import gc
    while 1:
        # testmodel = TestLeakModel()
        print('loop one')
        gc.collect()
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
