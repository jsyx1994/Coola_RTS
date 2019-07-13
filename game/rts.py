import numpy as np
from game.self_defined_actions import WorkerAction
from game.setttings import units_type

FREE = 0
WALL = 1
ALLY = 2
ENEMY = 3
RESOURCE = 4
BASE = 5


class Unit:
    def __init__(self, unit):
        self.type = unit['type']
        self.id = unit['ID']
        self.player = unit['player']
        self.x = unit['x']
        self.y = unit['y']
        self.resources = unit['resources']
        self.hitpoints = unit['hitpoints']


class RtsUtils:
    def __init__(self):

        # np.set_printoptions(threshold=np.inf)
        self.player_actions = ''
        self.last_hp_self = 0    # last step's sum of hps of self
        self.last_hp_oppo = 0    # last step's sum of hps of opponent

        self.gs = None
        self.player = None
        self.game_map = None    # used to translate the actions
        self.state = None       # state without location
        self.first = True       # flag to form the player_actions
        self._busy = None

    def reset_reward(self):
        self.last_hp_oppo = 0
        self.last_hp_self = 0

    def reset_game_state(self, gs: dict, player: int):
        """
        :param gs: game state dict from java end
        :param player: current player
        """
        self.player_actions = '['
        self.gs = gs
        self.player = player
        self.game_map = None  # used to translate the actions
        self.state = None  # state without location
        self.first = True
        self._busy = self.get_self_busy()
        self.construct_game_map()
        # print('hp', self.get_reward_test())

    def get_player(self):
        return self.player

    @property
    def is_game_over(self):
        return self.gs['gameOver']

    def get_winner(self):
        return self.gs['winner']

    def get_rl_bundle(self):
        return self.parse_game_state(), self.get_reward(), self.is_game_over

    def construct_game_map(self):
        # construct the game_map
        pgs = self.gs['pgs']
        self.game_map = self._parse_terrain()
        # print(self.game_map)
        units = pgs['units']
        for unit in units:
            _player = unit['player']
            y = unit['x']  # in array, row index first, but in the game the column first
            x = unit['y']
            if _player == -1:
                self.game_map[x][y] = RESOURCE  # resources
            elif self.player == _player:
                self.game_map[x][y] = ALLY  # allies
                if unit['type'] == 'Base':
                    self.game_map[x][y] = BASE  # my Base
            else:
                self.game_map[x][y] = ENEMY  # enemies
        # print(self.game_map)  # human readable game map
        self.game_map = self.game_map.transpose((1, 0))  # transpose to correspond to later manipulating

    def get_reward(self):
        pgs = self.gs['pgs']
        units = pgs['units']
        hp_self = 0
        hp_oppo = 0
        for unit in units:
            _player = unit['player']
            if self.player == _player:
                hp_self += int(unit['hitpoints'])
            elif _player == -1:
                pass
            else:
                hp_oppo += int(unit['hitpoints'])
        # print(self.player, hp_self, hp_oppo)
        delta_hp_self = hp_self - self.last_hp_self
        delta_hp_oppo = hp_oppo - self.last_hp_oppo
        reward_self = delta_hp_self - delta_hp_oppo
        self.last_hp_self, self.last_hp_oppo = hp_self, hp_oppo
        # if reward_self > 0:
        #     print(reward_self)
        if self.get_winner() == self.player:
            reward_self += 100
        # else:
        #     return 0
        return reward_self

    def get_self_units(self):
        """
        :return: list of our units
        """
        pgs = self.gs['pgs']
        return [unit for unit in pgs['units'] if unit['player'] == self.player]


    def _parse_terrain(self) -> np.ndarray:
        """
        parse the game state's terrain from java end to WALL and FREE
        :return:  terrain array info
        """
        gs = self.gs
        pgs = gs['pgs']
        terrain = pgs['terrain']
        height = pgs['height']
        width = pgs['width']
        game_map = np.full((width, height), FREE)
        for i, t in enumerate(terrain):
            if t == '1':
                x = i // width
                y = i - x * width
                game_map[x][y] = WALL
        return game_map

    def parse_game_state(self):
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
        channel_enemy_type = spatial_features[10:16]
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

    def add_location(self, x: int, y: int) -> np.ndarray:
        """
        :param x: x axis
        :param y: y axis
        :return: complete spatial features
        """
        spatial_features = self.state
        shape = (spatial_features[0].shape)
        channel_location = np.zeros((1, shape[0], shape[1]))
        channel_location[0][x][y] = 1
        spatial_features = np.concatenate([spatial_features, channel_location])
        return spatial_features

    def get_assignable(self):
        """
        get assignable units, busy units are excluded
        :return: unit list
        """
        # units = []
        pgs = self.gs['pgs']
        # # print('busy:', _busy)
        # for unit in pgs['units']:
        #     # get all the units of current player
        #     if (unit['player'] == self.player) and (unit['ID'] not in self._busy):
        #         units.append(unit)
        return [unit for unit in pgs['units'] if (unit['player'] == self.player) and (unit['ID'] not in self._busy)]

    def is_assignable(self, unit):
        return (unit['player'] == self.player) and (unit['ID'] not in self._busy)

    def get_actions(self):
        actions = self.gs['actions']
        return [action for action in actions]

    def get_action_type(self, action):
        return action['action']['type']


    def get_self_busy(self):
        """
        :return: busy ID list
        """
        self_units = self.get_self_units()
        self_units_id = [unit['ID'] for unit in self_units]
        actions = self.gs['actions']
        # units = self.gs['pgs']['units']
        _busy = [action['ID'] for action in actions if (action['action']['type'] != 0) and (action['ID'] in self_units_id)]

        # _busy_units = [unit for unit in units if unit['ID'] in _busy]
        # _busy_actions = [action for action in actions if action['ID'] in _busy]
        return _busy

    def _add_player_action(self, json_action: str):
        if self.first:
            self.player_actions += json_action
            self.first = False
        else:
            self.player_actions += ',' + json_action

    def get_player_action(self) -> str:
        return self.player_actions + ']'

    def translate_action(self, uid=-1, location=None, bot_type=None, act_code=0):
        """
        :param uid: unit id
        :param bot_type:
        :param act_code: self-defined actions' code
        :param location: locations
        :return:
        """
        assert bot_type in units_type

        if bot_type == 'Worker':
            worker_action = WorkerAction(self.game_map)
            json_action = worker_action.translate(uid, act_code, location)
            self._add_player_action(json_action)
        elif bot_type == 'Light':
            raise NotImplementedError
        elif bot_type == 'Heavy':
            raise NotImplementedError
        elif bot_type == 'Ranged':
            raise NotImplementedError
        elif bot_type == 'Base':
            raise NotImplementedError
        elif bot_type == 'Barracks':
            raise NotImplementedError


if __name__ == '__main__':
    gs = {'time': 0, 'pgs': {'width': 32, 'height': 32, 'terrain': '0000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000011111000000001111100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000', 'players': [{'ID': 0, 'resources': 20}, {'ID': 1, 'resources': 20}], 'units': [{'type': 'Resource', 'ID': 8, 'player': -1, 'x': 30, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 9, 'player': -1, 'x': 31, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 10, 'player': -1, 'x': 30, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 11, 'player': -1, 'x': 31, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 12, 'player': -1, 'x': 0, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 13, 'player': -1, 'x': 1, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 14, 'player': -1, 'x': 0, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 15, 'player': -1, 'x': 1, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 20, 'player': -1, 'x': 2, 'y': 31, 'resources': 10, 'hitpoints': 1}, {'type': 'Resource', 'ID': 21, 'player': -1, 'x': 0, 'y': 29, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 22, 'player': -1, 'x': 29, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 23, 'player': -1, 'x': 31, 'y': 2, 'resources': 20, 'hitpoints': 1}, {'type': 'Base', 'ID': 25, 'player': 0, 'x': 6, 'y': 14, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 26, 'player': 0, 'x': 6, 'y': 12, 'resources': 0, 'hitpoints': 1}, {'type': 'Base', 'ID': 27, 'player': 1, 'x': 25, 'y': 17, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 28, 'player': 1, 'x': 25, 'y': 19, 'resources': 0, 'hitpoints': 1}]}, 'actions': []}
    # print(gs)
    # gs = json.loads(gs)
    rts_utils = RtsUtils()
    rts_utils.reset(gs, 0)
    print(rts_utils.get_self_units())
