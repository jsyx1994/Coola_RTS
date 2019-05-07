import numpy as np
import json

FREE = 0
WALL = 1
ALLY = 2
ENEMY = 3
RESOURCE = 4


class UnitActions:
    """
    translating from java
    """
    TYPE_NONE = 0
    TYPE_MOVE = 1
    TYPE_HARVEST = 2
    TYPE_RETURN = 3
    TYPE_PRODUCE = 4
    TYPE_ATTACK_LOCATION = 5
    TYPE_NUMBER_OF_ACTION_TYPES = 6

    DIRECTION_NONE = -1
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    DIRECTION_OFFSET_X = [0, 1, 0, -1]
    DIRECTION_OFFSET_Y = [-1, 0, 1, 0]
    DIRECTION_NUMBER = 4


class Action:
    def __init__(self, game_map):
        self.game_map = game_map
        self.json_act = '{"unitID": "", "unitAction":{"type":"", "parameter": -1, "x":-1,"y":-1, "unitType":""}}'
        self.dict_action = json.loads(self.json_act)

    def translate(self, uid, act_code, location):
        assert isinstance(uid, int)
        assert isinstance(act_code, int)

    def probe(self,uid, x, y, dir):
        action = self.dict_action
        unit_action = action['unitAction']
        x += UnitActions.DIRECTION_OFFSET_X[dir]
        y += UnitActions.DIRECTION_OFFSET_Y[dir]

        action['unitID'] = uid
        unit_action['parameter'] = dir
        unit_action['x'] = x
        unit_action['y'] = y

        if self.game_map[x][y] == FREE:
            unit_action['type'] = UnitActions.TYPE_MOVE
            # action.format(uid, UnitActions.TYPE_MOVE, dir)
        elif self.game_map[x][y] == ENEMY:
            unit_action['type'] = UnitActions.TYPE_ATTACK_LOCATION
            # action.format(uid, UnitActions.TYPE_ATTACK_LOCATION, dir, x, y)
        elif self.game_map[x][y] == RESOURCE:
            unit_action['type'] = UnitActions.TYPE_HARVEST
            # action.format(uid, UnitActions.TYPE_HARVEST, dir, x, y)
        elif self.game_map[x][y] in (ALLY, WALL):
            unit_action['type'] = UnitActions.TYPE_NONE
        print(action)
        return action


class WorkerAction(Action):
    """
    define the DIY actions
    """
    NO_OP = 0
    UP = 1  # including moving and attacking behaviour
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    BUILD_BASE = 5
    BUILD_BARRACK = 6

    NUMBER_OF_ACTIONS = 7

    def __init__(self, game_map):
        super(WorkerAction, self).__init__(game_map)

    def translate(self, uid: int, act_code: int, location: tuple) -> str:
        """
        :param uid: id in real game
        :param act_code: self-defined code. e.g. NO_OP
        :param location:
        :return: json style unit action
        """
        super(WorkerAction, self).translate(uid, act_code, location)
        action = self.dict_action
        unit_action = action['unitAction']

        action['unitID'] = uid
        x, y = location
        if act_code == WorkerAction.NO_OP:
            unit_action['type'] = UnitActions.TYPE_NONE
        elif act_code == WorkerAction.UP:
            dir = UnitActions.DIRECTION_UP
            action = self.probe(uid, x, y, dir)

        elif act_code == WorkerAction.RIGHT:
            dir = UnitActions.DIRECTION_RIGHT
            action = self.probe(uid, x, y, dir)

        elif act_code == WorkerAction.DOWN:
            dir = UnitActions.DIRECTION_DOWN
            action = self.probe(uid, x, y, dir)

        elif act_code == WorkerAction.LEFT:
            dir = UnitActions.DIRECTION_LEFT
            action = self.probe(uid, x, y, dir)

        elif act_code == WorkerAction.BUILD_BASE:
            act = UnitActions.TYPE_PRODUCE
            dir = np.random.randint(4)
            unit_action['type'] = act
            unit_action['unitType'] = 'Base'
            unit_action['parameter'] = dir
            # json_act = json_act.format(uid, act, dir, -1, -1, 'Base')

        elif act_code == WorkerAction.BUILD_BARRACK:
            act = UnitActions.TYPE_PRODUCE
            dir = np.random.randint(4)
            unit_action['type'] = act
            unit_action['unitType'] = 'Barracks'
            unit_action['parameter'] = dir

        return json.dumps(action)


if __name__ == '__main__':
    pass






