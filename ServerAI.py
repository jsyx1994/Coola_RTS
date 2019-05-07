import socket
import json
from utils.rts import RtsUtils
from ai.bot import Bot
# @author: Jiawei, Y.
# Main file to look at.
# Methods you should look at:
#       BabyAI.getAction(player, gs)
#       policy(player, gs)
# Currently, policy always returns "Do nothing until be killed."
# For the details "parameter" and "type" in the return dict of 
# policy, please check hardCodedJSON.py

class ServerAI:
    """
    Python Version server.

    This is nothing more than a simple python implementation of ``Runserverexample.java``.

    serverAI creates a socket to listen to the given address and call ``SocketWrapperAI``
    for handling communication.

    Parameters
    ----------
    ``serverSocket`` : socket, optional default None.
        The serverSocket for channeling.
        Initialize the server socket by this parameters.
    
    ``host`` : string, optional default '127.0.0.1'.
    
    ``port`` : int, optional default '9898'.

    ``ai`` : object, optional default None.
        The ``ai`` used in server side.

    ``DEBUG`` : int, optional default '1'.
        The debug parameter.
        Print intermediate process when set to '1'
    """

    DEBUG = 1

    def __init__(self, server_socket=None, host='127.0.0.1', port=9898, ai=None):
        self.socket_addr = (host, port)
        self.ai = ai
        if server_socket is None:    # issue 1
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.server_socket = server_socket

    def runServer(self):
        """
        run the server

        Parameters
        ----------
        self: object

        Returns
        -------
        None
        """
        client_number = 0    # issue 2 :self-play situation
        self.server_socket.bind(self.socket_addr)
        self.server_socket.listen(10)
        if self.DEBUG >= 1:
            print('Server is running')
        try:
            while True:
                print('waiting for a connection')
                client_socket, client_address = self.server_socket.accept()
                SocketWrapperAI(client_socket, client_number, self.ai)
        finally:
            self.server_socket.close()


class SocketWrapperAI:
    """
    Python Version SocketWrapperAI.

    SocketWrapperAI handle with welcome messages and acknowledgements between clients and server, 
    as well as passing the game state information to the ``ai`` used in this server.

    Parameters
    ----------
    ``DEBUG``: int, optional default '1'.
        The debug parameter.
        Print intermediate process when set to '1'
    
    ``utt``: object, optional default None.
        ``UnityTypeTable``

    ``ai``: object, optional default None.
        ``ai``

    All of the remaining parameters will be passed through serverAI, don't worry it.
    
    Notes
    ---------
    This is nothing more than a simple python implementation of ``JSONRunserverexample.java``.
    However, some internal implementation details is different from santi's one.
    """

    DEBUG = 1

    def __init__(self, client_socket, client_number, ai=None):
        self.client_socket = client_socket
        self.client_number = client_number
        # utt: UnityTypeTable
        self.utt = None
        self.ai = ai
        if self.DEBUG >= 1:
            print("New connection with client# {} at {}".format(client_number, client_socket))
        self.run()

    def run(self):
        """
        If you don't understand the main logic behind santi's work, 
        please carefully go through the comments of this function.
        """
        try:
            welcome = 'PyJSONSocketWrapperAI: you are client #%i\n' % self.client_number
            self.client_socket.sendall(welcome.encode())
            while True:
                # retrieve the message from socket
                msg = str(self.client_socket.recv(10240).decode())
                # print(msg)
                # connection broken, retry.
                if msg == 0:
                    break

                # client closes the connection.
                elif msg.startswith('end'):
                    self.client_socket.close()
                    exit(self)

                # budget initialization
                elif msg.startswith('budget'):
                    msg = msg.split()
                    self.ai.reset()
                    self.ai.timeBudget = int(msg[1])
                    self.ai.iterationsBudget = int(msg[2])
                    if self.DEBUG >= 1:
                        print("setting the budget to: {},{}".format(self.ai.timeBudget, self.ai.iterationsBudget))
                    # send back ack
                    self.client_socket.sendall('ack\n'.encode())

                    # utt initialization
                elif msg.startswith('utt'):

                    msg = msg.split('\n')[1]

                    # json.loads() returns dict
                    self.utt = json.loads(msg)

                    # reset ai with current utt
                    self.ai.reset(self.utt)
                    if self.DEBUG >= 1:
                        print('setting the utt to: {}'.format(self.utt))
                        pass

                    # send back ack
                    self.client_socket.sendall('ack\n'.encode())

                    # client asks server to return units actions for current game state
                elif msg.startswith('getAction'):
                    # get player ID
                    player = int(msg.split()[1])  ### issue
                    # get game state
                    gs = msg.split('\n')[1]
                    # json.loads() returns dict
                    gs = json.loads(gs)
                    rts_utils = RtsUtils(gs, player)
                    assignable = rts_utils.get_assignable()
                    import torch
                    if gs['time'] % 100 == 0:
                        for unit in assignable:
                            if unit['type'] == 'Worker':
                                bot = Bot(bot_type='worker')
                                rts_utils.translate_action(uid=unit['ID'], location=(int(unit['x']), int(unit['y'])),
                                                           bot_type='worker', act_code=bot.decide(torch.randn(1, 20, 8, 8)))

                    # print(rts_utils.get_assignable())
                    pa = rts_utils.get_player_action()
                    if self.DEBUG >= 1:
                        print('getAction for player {}'.format(player))
                        print('with game state %s' % gs)

                    # call ai.getAction to return the player actions ##string##
                    # pa = self.ai.get_action(player, gs)
                    # pa = '[]'
                    # example
                    # if gs['time'] % 20 == 0:
                    #     pa = '[{"unitID":22,"unitAction":{"type":1,"parameter":2,"unitType":""}}]'
                    # send the encoded player actions string to client
                    self.client_socket.sendall(('%s\n' % pa).encode())

                    if self.DEBUG >= 1:
                        print('action sent!')

                # get preGameAnalysis
                elif msg.startswith('preGameAnalysis'):
                    print('get preGameAnalysis, it has not been implemented yet')
                    pass

                elif msg.startswith('gameOver'):
                    msg = msg.split()
                    winner = msg[1]
                    if self.DEBUG >= 1:
                        print('gameOver %s' % winner)
                    self.ai.game_over(winner)
                    self.client_socket.sendall(('ack\n').encode())
                    self.client_socket.close()
        finally:
            self.client_socket.close()
            print('Connection with client# {} closed'.format(self.client_number))


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
    serverAI = ServerAI(ai=babyAI)
    serverAI.runServer()
