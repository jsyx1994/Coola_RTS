import torch
from shared_models.model import ActorCritic
from torch.distributions import Categorical
from game.rts import RtsUtils
from game.setttings import units_type

class Bot:
    def __init__(self, bot_type=None, bot_id=0):
        assert isinstance(bot_id, int)
        assert bot_type in units_type

        self.model = ActorCritic(actor=bot_type)
        # print(self.model)
        # try:
        #     self.model.load_state_dict(torch.load('../shared_models/{}.pt'.format(bot_type)))
        # except FileNotFoundError as e:
        #     print(e)

        self.type = bot_type

    def predict(self, state):
        """
        forwarding the model
        :param state: game state M * N * f
        :return: the action code chosen
        """
        return self.model.forward(state)

    def decide(self, state, mode='sample'):
        prediction = self.predict(state)
        if mode is 'sample':
            m = Categorical(prediction[0])
            action = m.sample()
            print(m, action)
            return int(action[0])


if __name__ == '__main__':
    gs = {'time': 1600, 'pgs': {'width': 32, 'height': 32, 'terrain': '0000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000011111000000001111100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000', 'players': [{'ID': 0, 'resources': 5}, {'ID': 1, 'resources': 20}], 'units': [{'type': 'Resource', 'ID': 8, 'player': -1, 'x': 30, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 9, 'player': -1, 'x': 31, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 10, 'player': -1, 'x': 30, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 11, 'player': -1, 'x': 31, 'y': 1, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 12, 'player': -1, 'x': 0, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 13, 'player': -1, 'x': 1, 'y': 30, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 14, 'player': -1, 'x': 0, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 15, 'player': -1, 'x': 1, 'y': 31, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 20, 'player': -1, 'x': 2, 'y': 31, 'resources': 10, 'hitpoints': 1}, {'type': 'Resource', 'ID': 21, 'player': -1, 'x': 0, 'y': 29, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 22, 'player': -1, 'x': 29, 'y': 0, 'resources': 20, 'hitpoints': 1}, {'type': 'Resource', 'ID': 23, 'player': -1, 'x': 31, 'y': 2, 'resources': 20, 'hitpoints': 1}, {'type': 'Base', 'ID': 25, 'player': 0, 'x': 6, 'y': 14, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 26, 'player': 0, 'x': 3, 'y': 10, 'resources': 0, 'hitpoints': 1}, {'type': 'Base', 'ID': 27, 'player': 1, 'x': 25, 'y': 17, 'resources': 0, 'hitpoints': 10}, {'type': 'Worker', 'ID': 28, 'player': 1, 'x': 25, 'y': 19, 'resources': 0, 'hitpoints': 1}, {'type': 'Barracks', 'ID': 29, 'player': 0, 'x': 4, 'y': 10, 'resources': 0, 'hitpoints': 4}, {'type': 'Base', 'ID': 30, 'player': 0, 'x': 2, 'y': 10, 'resources': 0, 'hitpoints': 10}]}, 'actions': []}
    # print(Bot(bot_type='Worker').predict(torch.randn(1, 20, 8, 8)))
    rts_utils = RtsUtils(gs, 0)
    assignable = rts_utils.get_assignable()
    bot = Bot(bot_type='Worker')
    for unit in assignable:
        if unit['type'] == 'Worker':
            rts_utils.translate_action(uid=unit['ID'], location=(int(unit['x']), int(unit['y'])), bot_type='Worker', act_code=bot.decide(torch.randn(1, 20, 8, 8)))
    print(rts_utils.get_player_action())
    # print(Bot(bot_type='worker').predict(torch.randn(1, 20, 8, 8))
