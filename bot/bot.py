import random
from sys import stderr

from move.move import Move
from move.move_type import MoveType

from game_state import GOLADState
from mcts import UCT

from config import config

class Bot:
    """
    Base class for bots
    """        
    def __init__(self):
        random.seed()  # set seed here if needed

    def make_move(self, game):
        """
        Given a Game object, return a Move object
        """
        return Move(MoveType.PASS)

class MctsBot(Bot):

    def __init__(self, nn):
        random.seed()  # set seed here if needed
        self.nn = nn
        
    def make_move(self, game):
        """
        Performs a Birth or a Kill move, currently returns a random move.
        Implement this method to make the bot smarter.
        """
        file = open('errorlog.txt', 'a+')
        file.write('making a move\n')
        file.close()
        
        state = GOLADState(field=game.field)
        state.current_player = int(game.me.id)
        c, pi= UCT(rootstate = state, itermax = config.mcts_itermax, nn=self.nn, verbose = True)

        return c.move

class RandomBot(Bot):

    def __init__(self):
        random.seed()  # set seed here if needed

    def get_action(self, game):
        cell_map = game.field.get_cell_mapping()
        dead_cells = cell_map.get('.', [])
        my_cells = cell_map.get(game.me.id, [])
        opponent_cells = cell_map.get(game.opponent.id, [])
        living_cells = my_cells + opponent_cells

        action = random.randrange(3)
        random_death = living_cells[random.randrange(len(living_cells))]
        if len(dead_cells) > 0 and len(my_cells) >= 2:
            random_birth = dead_cells[random.randrange(len(dead_cells))]
            random_sacrifices = [my_cells.pop(random.randrange(len(my_cells))) for x in range(2)]
        else:
            random_birth = living_cells[0]

        return action, [random_death] + [random_birth] + random_sacrifices

    def make_move(self, game):
        """
        Performs a Birth or a Kill move, currently returns a random move.
        Implement this method to make the bot smarter.
        """

        cell_map = game.field.get_cell_mapping()

        if random.random() < 0.5:
            return self.make_random_birth_move(game, cell_map)

        return self.make_random_kill_move(game, cell_map)

#         action, cells = self.get_action(game)
#         if action == 0:
#             return Move(MoveType.KILL, cells[0])
#         elif action == 1:
#             return Move(MoveType.BIRTH, cells[1], cells[2:])
#         else:
#             return Move(MoveType.PASS)

    def make_random_birth_move(self, game, cell_map):
        dead_cells = cell_map.get('.', [])
        my_cells = cell_map.get(game.me.id, [])

        if len(dead_cells) <= 0 or len(my_cells) < 2:
            return self.make_random_kill_move(game, cell_map)

        random_birth = dead_cells[random.randrange(len(dead_cells))]

        random_sacrifices = []
        for i in range(2):
            random_sacrifice = my_cells.pop(random.randrange(len(my_cells)))
            random_sacrifices.append(random_sacrifice)

        return Move(MoveType.BIRTH, random_birth, random_sacrifices)

    def make_random_kill_move(self, game, cell_map):
        my_cells = cell_map.get(game.me.id, [])
        opponent_cells = cell_map.get(game.opponent.id, [])
        living_cells = my_cells + opponent_cells

        if len(living_cells) <= 0:
            return Move(MoveType.PASS)

        random_kill = living_cells[random.randrange(len(living_cells))]

        return Move(MoveType.KILL, random_kill)

class PassBot(Bot):

    def __init__(self):
        random.seed()  # set seed here if needed

    def make_move(self, game):
        return Move(MoveType.PASS)

