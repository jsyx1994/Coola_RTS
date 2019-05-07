import numpy as np
from utils.self_defined_actions import WorkerAction

FREE = 0
WALL = 1
ALLY = 2
ENEMY = 3
RESOURCE = 4


class RtsUtils:
    def __init__(self, gs, player):
        np.set_printoptions(threshold=np.inf)
        self.gs = gs
        self.player = player
        self.game_map = None    # used to translate the actions
        self.state = None   # state without location
        self.player_actions = '['
        self.first = True
        # construct the game_map
        pgs = gs['pgs']

        self.game_map = self._parse_terrain()
        # print(self.game_map)

        units = pgs['units']
        for unit in units:
            _player = unit['player']
            y = unit['x']   # in array, row index first, but in the game the column first
            x = unit['y']
            if _player == -1:
                self.game_map[x][y] = RESOURCE  # resources
            elif player == _player:
                self.game_map[x][y] = ALLY  # allies
            else:
                self.game_map[x][y] = ENEMY  # enemies
        print(self.game_map)
        self.game_map = self.game_map.transpose((1, 0))     # transpose to correspond to later manipulating


    def _parse_terrain(self):
        gs = self.gs
        pgs = gs['pgs']
        terrain = pgs['terrain']
        height = pgs['height']
        width = pgs['width']
        game_map = np.full((width, height), FREE)
        for i, t in enumerate(terrain):
            # print(t)
            if t == '1':
                x = i // width
                y = i - x * width
                game_map[x][y] = WALL
        return game_map

    def parse_game_state(self):
        """
            Parameters
            ----------
            ``current_player`` : int.
                Denotes current player.
            ``gs`` : dict, Game state data. json.loads()
                {'time': int, 'pgs':{...}}
            Returns
            -------
            ``spatial_features`` : A numpy arrary of shape(18,8,8)

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
            """
        gs = self.gs
        current_player = self.player

        # Used for type indexing
        utt = ['Base', 'Barracks', 'Worker', 'Light', 'Heavy', 'Ranged']
        type_idx = {}
        for i, ut in zip(range(len(utt)), utt):
            type_idx[ut] = i

        time = gs['time']
        pgs = gs['pgs']
        actions = gs['actions']
        w = pgs['width']
        h = pgs['height']
        units = pgs['units']

        # Initialization of spatial features
        spatial_features = np.zeros((18, h, w))

        # channel_wall
        spatial_features[0] = np.array([int(x) for x in pgs['terrain']]).reshape((1, h, w))

        # other channels
        channel_resource = spatial_features[1]
        channel_self_type = spatial_features[2:8]
        channel_self_hp = spatial_features[8]
        channel_self_resource_carried = spatial_features[9]
        channel_enemy_type = spatial_features[9:15]
        channel_enemy_hp = spatial_features[16]
        channel_enemy_resource_carried = spatial_features[17]

        for unit in units:
            _player = unit['player']
            x = unit['x']
            y = unit['y']
            # neutral
            if _player == -1:
                channel_resource[x][y] = unit['resources']

            elif _player == current_player:
                # get the index of this type
                idx = type_idx[unit['type']]
                channel_self_type[idx][x][y] = 1
                channel_self_hp[x][y] = unit['hitpoints']
                channel_self_resource_carried[x][y] = unit['resources']

            else:
                idx = type_idx[unit['type']]
                channel_enemy_type[idx][x][y] = 1
                channel_enemy_hp[x][y] = unit['hitpoints']
                channel_enemy_resource_carried[x][y] = unit['resources']

        # print(spatial_features)
        # print(spatial_features.shape)
        self.state = spatial_features
        return spatial_features

    def add_location(self, x, y):
        spatial_features = self.state
        shape = (spatial_features[0].shape)
        channel_location = np.zeros((1, shape[0], shape[1]))
        channel_location[0][x][y] = 1
        spatial_features = np.concatenate([spatial_features, channel_location])
        return spatial_features

    def get_assignable(self):
        """
        get assignable actions
        :return:
        """
        units = []
        gs = self.gs
        pgs = gs['pgs']
        actions = gs['actions']
        _busy = [action['ID'] for action in actions if action['action']['type'] != 0]
        print('busy:', _busy)
        for unit in pgs['units']:
            # get all the units of current player
            # print(unit['player'])
            if (unit['player'] == self.player) and (unit['ID'] not in _busy):
                units.append(unit)

        return units

    def _add_player_action(self, json_action):
        if self.first:
            self.player_actions += json_action
            self.first = False
        else:
            self.player_actions += ',' + json_action

    def get_player_action(self):
        return self.player_actions + ']'

    def translate_action(self, uid=-1, location=None, bot_type=None, act_code=0):
        """
        :param uid:
        :param bot_type:
        :param act_code:
        :param location:
        :return:
        """
        assert bot_type in (
            'worker',
            'light',
            'heavy',
            'ranged',
            'base',
            'barrack',
        )

        if bot_type == 'worker':
            worker_action = WorkerAction(self.game_map)
            json_action = worker_action.translate(uid, act_code, location)
            self._add_player_action(json_action)
        elif bot_type == 'light':
            pass
        elif bot_type == 'heavy':
            pass
        elif bot_type == 'ranged':
            pass
        elif bot_type == 'base':
            pass
        elif bot_type == 'barrack':
            pass



if __name__ == '__main__':
    gs = {'time': 0, 'pgs': {'width': 32, 'height': 32, 'terrain': '0000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000011111000000001111100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000', 'players': [{'ID': 0, 'resources': 20}, {'ID': 1, 'resources': 20}], 'units': [{'type': 'Resource', 'ID': 8, 'player': -1, 'x': 30, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 9, 'player': -1, 'x': 31, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 10, 'player': -1, 'x': 30, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 11, 'player': -1, 'x': 31, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 12, 'player': -1, 'x': 0, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 13, 'player': -1, 'x': 1, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 14, 'player': -1, 'x': 0, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 15, 'player': -1, 'x': 1, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 20, 'player': -1, 'x': 2, 'y': 31, 'resources': 10, 'hitpoints': 1}, {'type': 'Resource', 'ID': 21, 'player': -1, 'x': 0, 'y': 29, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 22, 'player': -1, 'x': 29, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 23, 'player': -1, 'x': 31, 'y': 2, 'resources': 20, 'hitpoints': 1}, {'type': 'Base', 'ID': 25, 'player': 0, 'x': 6, 'y': 14, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 26, 'player': 0, 'x': 6, 'y': 12, 'resources': 0, 'hitpoints': 1}, {'type': 'Base', 'ID': 27, 'player': 1, 'x': 25, 'y': 17, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 28, 'player': 1, 'x': 25, 'y': 19, 'resources': 0, 'hitpoints': 1}]}, 'actions': []}
    # print(gs)
    # gs = json.loads(gs)
    rts_utils = RtsUtils(gs, 0)
