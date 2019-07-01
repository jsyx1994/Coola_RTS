from game.rts import RtsUtils
import socket
import json

class Environment:
    def __init__(self):
        self.rts_utils = RtsUtils()

    def reset(self, gs, player):
        self.rts_utils.reset_game_state(gs, player)
        return self.rts_utils.get_rl_bundle()

    def parse_game_state(self):
        return self.rts_utils.parse_game_state()

    def step(self, sock: socket.socket, pa: str, player):
        sock.sendall(('%s\n' % pa).encode())
        print('action sent!')
        msg = str(sock.recv(10240).decode())
        if msg == 0:
            print('break')
            raise Exception
        gs = msg.split('\n')[1]
        gs = json.loads(gs)
        self.rts_utils.reset_game_state(gs,player)
        return self.rts_utils.get_rl_bundle()
