import socket
import json
from game.rts import RtsUtils
from multiprocessing import Process
from ai.bot import Bot
import torch
from shared_models.model import ActorCritic
from torch.distributions import Categorical
import numpy as np
from game.self_defined_actions import WorkerAction
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

rts_utils = RtsUtils()
worker = ActorCritic(actor='Worker')    # output  critic or worker policy
# class ServerAI:
#     """
#     Python Version server.
#
#     This is nothing more than a simple python implementation of ``Runserverexample.java``.
#
#     serverAI creates a socket to listen to the given address and call ``SocketWrapperAI``
#     for handling communication.
#
#     Parameters
#     ----------
#     ``serverSocket`` : socket, optional default None.
#         The serverSocket for channeling.
#         Initialize the server socket by this parameters.
#
#     ``host`` : string, optional default '127.0.0.1'.
#
#     ``port`` : int, optional default '9898'.
#
#     ``ai`` : object, optional default None.
#         The ``ai`` used in server side.
#
#     ``DEBUG`` : int, optional default '1'.
#         The debug parameter.
#         Print intermediate process when set to '1'
#     """
#
#     DEBUG = 1
#
#     def __init__(self, server_socket=None, host='127.0.0.1', port=9898, ai=None):
#         self.server_socket = server_socket
#         self.socket_addr = (host, port)
#         self.ai = ai
#
#
#         if server_socket is None:    # issue 1
#             self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         else:
#             self.server_socket = server_socket
#
#     def runServer(self):
#         """
#         run the server
#
#         Parameters
#         ----------
#         self: object
#
#         Returns
#         -------
#         None
#         """
#         client_number = 0    # issue 2 :self-play situation
#         self.server_socket.bind(self.socket_addr)
#         self.server_socket.listen(10)
#         if self.DEBUG >= 1:
#             print('Server is running')
#         try:
#             while True:
#                 print('waiting for a connection')
#                 client_socket, client_address = self.server_socket.accept()
#                 SocketWrapperAI(client_socket, client_number, self.ai)
#         finally:
#             print('server_socket closed')
#             self.server_socket.close()

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

                handler = SocketWrapperAI(conn, self.client_num)
                p = Process(target=handler.run_episodes, args=())
                p.start()


class SocketWrapperAI:
    DEBUG = 1
    GAMMA = 0.99

    def __init__(self, client_socket, client_number, ai=None):
        self.client_socket = client_socket
        self.client_number = client_number
        # utt: UnityTypeTable
        self.utt = None
        self.ai = ai
        if self.DEBUG >= 1:
            print("New connection with client# {} at {}".format(client_number, client_socket))

        self.reward = []
        self.targets = []
        self.predicts = []
        self.log_pi_sa = []
        self.p_state = None
        self.G_0s = []
        self.G_0 = 0


        self.worker_sharing_counts = 0

    def step(self, action):
        pass

    def init_new_episode(self):
        self.reward = []
        self.targets = []
        self.predicts = []

        self.log_pi_sa = []
        self.p_state = None
        self.G_0 = 0


        self.worker_sharing_counts = 0


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
                # print(msg)
                # connection broken, retry.
                if msg == 0:
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

                    state = torch.from_numpy(rts_utils.parse_game_state()).float().unsqueeze(0)     # s(t)
                    rt_1 = rts_utils.get_last_reward()
                    vst = worker.forward(state, info='critic')     # v(s_t)
                    self.G_0 += rt_1

                    # calculate targets and predicts

                    # print(self.p_state)
                    if self.p_state is not None:
                        vst_1 = worker.forward(self.p_state, info='critic')
                        target = rt_1 + self.GAMMA * vst_1
                        predict = vst
                        # print(self.worker_sharing_counts)
                        for _ in range(self.worker_sharing_counts):
                            self.targets.append(target)
                            self.predicts.append(predict)
                        # print(len(self.targets))
                        # print(len(self.predicts))
                        # print(len(self.log_pi_sa))

                    self.worker_sharing_counts = 0

                    if gs['time'] % 10 == 0:
                        for unit in assignable:
                            location = int(unit['x']), int(unit['y'])
                            if unit['type'] == 'Worker':
                                # print(worker.forward(state, loc=location))
                                pi = worker.forward(state, info=location)

                                action = int(Categorical(pi).sample()[0])   # a(t)
                                rts_utils.translate_action(uid=unit['ID'], location=location,
                                                           bot_type='Worker', act_code=action)
                                self.log_pi_sa.append(torch.log(pi[0][action]).view(1))
                                self.worker_sharing_counts += 1

                    # print(rts_utils.get_assignable())
                    pa = rts_utils.get_player_action()
                    if self.DEBUG >= 2:
                        print('getAction for player {}'.format(player))
                        print('with game state %s' % gs)

                    client_socket.sendall(('%s\n' % pa).encode())

                    self.p_state = state    # assign to previous states

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

                    self.G_0s.append(self.G_0)
                    self.plot_durations()
                    self.optimize()
                    # self.client_socket.close()
        # finally:
        #     # self.client_socket.close()
        #     print('Connection with client# {} closed'.format(self.client_number))

    def optimize(self):
        # print(self.log_pi_sa)
        optimizer = optim.Adam(params=worker.parameters(), lr=1e-3)
        log_pi_sa = torch.cat(self.log_pi_sa)
        targets = torch.cat(self.targets)
        predicts = torch.cat(self.predicts)
        # print(log_pi_sa.requires_grad)
        # print(targets.requires_grad)
        # print(predicts.requires_grad)
        # print(type(targets))
        pg_loss = log_pi_sa * (targets - predicts)
        td_error = F.mse_loss(targets, predicts)
        ac_loss = -pg_loss.mean() + td_error
        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
        print("optimizing done")

    @staticmethod
    def select_action(unit_nn, state):
        prediction = unit_nn.predict(state)
        m = Categorical(prediction[0])
        action = m.sample()
        print(m, action)
        return int(action[0])

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


class BabyAI:
    def __init__(self, utt=None, policy=None):
        self.timeBudget = 100
        self.iterationsBudget = 0
        self.utt = utt
        self.actions = {}
        self.policy = policy

    def reset(self, utt=None):
        self.utt = utt
        return

    def get_action(self, player, gs):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

  
        Parameters
        ----------
        ``player`` : int. 
            Denotes current player.

        ``gs`` : dict, Game state data.
             {'time': int, 'pgs':{...}}


        Returns
        -------
        ``msg`` : string, unitActions to send back.
            '["unitID":int, "unitAction":{"type": int, "parameter": int, "unitType": str}]'
            

        Examples
        --------
        ``gs``:

            {'time': 0, 
             'pgs': {'width': 8, 
                     'height': 8, 
                     'terrain': '0000000000000000000000000000000000000000000000000000000000000000', 
                     'players': [{'ID': 0, 'resources': 5}, 
                                 {'ID': 1, 'resources': 5}], 
                     'units':   [{'type': 'Resource', 'ID': 0, 'player': -1, 
                                  'x': 0, 'y': 0, 'resources': 20, 'hitpoints': 1
                                 }, 
                                 {'type': 'Resource',  'ID': 1,  'player': -1, 
                                  'x': 7,  'y': 7,  'resources': 20, 'hitpoints': 1
                                 }, 
                                 {'type': 'Base', 'ID': 2, 'player': 0, 
                                  'x': 2, 'y': 1, 'resources': 0, 'hitpoints': 10
                                 }, 
                                 {'type': 'Base', 'ID': 3, 'player': 1, 
                                  'x': 5, 'y': 6, 'resources': 0, 'hitpoints': 10
                                 }]}, 
             'actions': []
            }


        ``msg``: string

        [
            { "unitID":4, 
              "unitAction":{"type": 2, "parameter": 0}
            },
            { "unitID":6, 
              "unitAction":{"type": 1, "parameter": 1}
            },
            { "unitID":8, 
              "unitAction":{"type": 1, "parameter": 2}
            }
            { "unitID":2, 
              "unitAction":{"type":4, "parameter":0, "unitType":"Worker"}
        ]
        """
        msg = '['

        '''
        policy returns dict:
            {id:{type':int, 'isAttack':boolean, 'x':int, 'y':int, 'parameter':int, 'unitType':string}}

        Note: the return of policy differs from getAction
        '''
        self.actions = policy(player, gs)

        first = True
        for unit, unit_action in self.actions:
            if first == False:
                msg = msg + ' ,'
            if unit_action['isAttack'] == True:
                msg = msg + '{"unitID":{}, "unitAction":{"type":{}, "x":{},"y":{}}'.format(unit, unit_action['type'],
                                                                                           unit_action['x'],
                                                                                           unit_action['y'])
            else:
                msg = msg + '{"unitID":{}, "unitAction":{"type":{}, "parameter":{}, "unitType":"{}"}'.format(unit,
                                                                                                             unit_action[
                                                                                                                 'type'],
                                                                                                             unit_action[
                                                                                                                 'parameter'],
                                                                                                             unit_action[
                                                                                                                 'unitType'])

            first = False

        msg = msg + ']'
        return msg

    def game_over(self, winner):
        print('winner: ', winner)
        return


def policy(player, gs):
    """Policy used to generate actions.

  
        Parameters
        ----------
        ``player`` : int. 
            Denotes current player.

        ``gs`` : dict, Game state data.
             {'time': int, 'pgs':{...}}


        Returns
        -------
        ``unitsActions`` : dict, unitActions to send back.
            {id:{type':int, 'isAttack':boolean, 'x':int, 'y':int, 'parameter':int, 'unitType':string}}
            

        Examples
        --------
        ``gs``:

            {'time': 0, 
             'pgs': {'width': 8, 
                     'height': 8, 
                     'terrain': '0000000000000000000000000000000000000000000000000000000000000000', 
                     'players': [{'ID': 0, 'resources': 5}, 
                                 {'ID': 1, 'resources': 5}], 
                     'units':   [{'type': 'Resource', 'ID': 0, 'player': -1, 
                                  'x': 0, 'y': 0, 'resources': 20, 'hitpoints': 1
                                 }, 
                                 {'type': 'Resource',  'ID': 1,  'player': -1, 
                                  'x': 7,  'y': 7,  'resources': 20, 'hitpoints': 1
                                 }, 
                                 {'type': 'Base', 'ID': 2, 'player': 0, 
                                  'x': 2, 'y': 1, 'resources': 0, 'hitpoints': 10
                                 }, 
                                 {'type': 'Base', 'ID': 3, 'player': 1, 
                                  'x': 5, 'y': 6, 'resources': 0, 'hitpoints': 10
                                 }]}, 
             'actions': []
            }


        ``unitsActions``: dict
            { 3: {'type': 0, 'isAttack': False, 'x': 0, 'y': 0, 'parameter': 0, 'unitType':'Worker'},
              5: {'type': 1, 'isAttack': False, 'x': 0, 'y': 0, 'parameter': 2, 'unitType':'Worker'},
              7: {'type': 5, 'isAttack': True, 'x': 3, 'y': 8, 'parameter': 0, 'unitType':'Ranged'},
            }
    """
    # gs is a json string
    # get all the units of current player
    units = []
    enemyUnits = []
    resources = []
    unitsActions = {}
    tmpDic = {'type': 0, 'isAttack': False, 'x': 0, 'y': 0, 'parameter': 0, 'unitType': 'default'}

    for unit in gs['pgs']['units']:
        # get resources
        if unit['player'] == -1:
            resources.append(unit)
        else:
            # get all the units of current player
            if unit['player'] == player:
                units.append(unit)
                tmpDic['unitType'] = unit['type']

                # assign actions to all the units of current player
                unitsActions[unit['ID']] = tmpDic
            else:
                # get enemyUnits
                enemyUnits.append(unit)

    return unitsActions


if __name__ == "__main__":
    babyAI = BabyAI()
    serverAI = ServerAI()
    serverAI.run_server()

