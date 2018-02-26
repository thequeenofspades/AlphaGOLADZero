import itertools

from field.point import Point
from field.field import Field
from move.move import Move
from move.move_type import MoveType

import numpy as np

class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

class GOLADState(GameState):
    pass
    """ A state of the game of GOLAD, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 0 (0), 2 = player 1 (1).
    """
    def __init__(self, field):
        self.current_player = 0
        self.timestep = 0
        self.terminal = 0 # 0 (not done), 1 (player0 wins), 2 (player1 wins), 3 (tie)
        self.field = field
        

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GOLADState(self.field.Clone())
        st.current_player = self.current_player
        return st

    def Convert(self):
        # Convert the state to the state representation expected by the neural network in nn/nn.py
        cells = self.field.cells
        state = np.zeros((self.field.width, self.field.height, 3))
        for i in range(self.field.width):
            for j in range(self.field.height):
                if self.cells[i][j] == '0':
                    state[i,j,0] = 1
                elif self.cells[i][j] == '1':
                    state[i,j,1] = 1
                state[i,j,2] = self.current_player
        return state

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        # Apply cell change
        if move.move_type == MoveType.KILL:
            self.field.cells[move.target_point.x][move.target_point.y] = '.'
        elif move.move_type == MoveType.BIRTH:
            self.field.cells[move.target_point.x][move.target_point.y] = str(self.current_player)
            self.field.cells[move.sacrifice_points[0].x][move.sacrifice_points[0].y] = '.'
            self.field.cells[move.sacrifice_points[1].x][move.sacrifice_points[1].y] = '.'
        elif move.move_type == MoveType.PASS:
            pass

        # Simulate the game for 1 step
        cell_map = self.field.get_cell_mapping()
        dead_cells = cell_map.get('.', [])
        my_cells = cell_map.get(str(self.current_player), [])
        opp_cells = cell_map.get(str(1 - self.current_player), [])
        living_cells = my_cells + opp_cells

        new_field = self.field.Clone()
        for cell in living_cells:
            count = self.field.count_neighbors(cell.x, cell.y)
            if count[0] < 2 or count[0] > 3:
                new_field.cells[cell.x][cell.y] = '.'

        for cell in dead_cells:
            count = self.field.count_neighbors(cell.x, cell.y)
            if count[0] == 3:
                new_field.cells[cell.x][cell.y] = '0' if count[1]>count[2] else '1'

        self.field = new_field
        self.timestep += 1

        # Flip turn player
        self.current_player = 1 - self.current_player

        # Update self.terminal
        cell_map = self.field.get_cell_mapping()
        cells_0 = cell_map.get('0', [])
        cells_1 = cell_map.get('1', [])
        if (len(cells_0) > 0) and (len(cells_1) <= 0):
            self.terminal = 1
        elif (len(cells_0) <= 0) and (len(cells_1) > 0):
            self.terminal = 2
        elif self.timestep >= self.max_timestep:
            self.terminal = 3

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        moves = []
        if self.terminal != 0:
            return []
#         curr_player_cell = "0" if self.current_player==0 else "1"
#         cells_empty = []
#         cells_self = []
#         for i in range(self.width):
#             for j in range(self.height):
#                 if cell[i][j] == ".":
#                     cells_empty.append((i,j))
#                 elif cell[i][j] == curr_player_cell:
#                     cells_self.append((i,j))
        cell_map = self.field.get_cell_mapping()
        dead_cells = cell_map.get('.', [])
        my_cells = cell_map.get(str(self.current_player), [])
        opp_cells = cell_map.get(str(1 - self.current_player), [])
        living_cells = my_cells + opp_cells
        # Generate kill moves
        for kill_cell in living_cells:
            moves.append(Move(MoveType.KILL, kill_cell))
        # Generate birth moves
        for birth_cell in dead_cells:
            for sacrifice_cells in itertools.combinations(my_cells, 2):
                moves.append(Move(MoveType.BIRTH, birth_cell, [sacrifice_cells[0], sacrifice_cells[1]]))
        # Generate pass move
        moves.append(Move(MoveType.PASS))
        return moves

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.field.width and y >= 0 and y < self.field.height

    def CoordsToIndex(self, x, y):
        return x*self.field.width + y

    def GetP(self, net_probs, valid_moves):
        action_logits, birth_logits, sac_logits, kill_logits, _ = net_probs
        p = {}
        for move in valid_moves:
            if move.move_type == MoveType.BIRTH:
                p[move] = np.squeeze(action_logits)[0]  # assuming batch size of 1
                birth_cell = move.target_point
                sac_cell1, sac_cell2 = move.sacrifice_points
                p[move] = p[move] * np.squeeze(birth_logits)[self.CoordsToIndex(birth_cell.x, birth_cell.y)]
                p[move] = p[move] * np.squeeze(sac_logits)[self.CoordsToIndex(sac_cell1.x, sac_cell1.y)]
                p[move] = p[move] * np.squeeze(sac_logits)[self.CoordsToIndex(sac_cell2.x, sac_cell2.y)]
            elif move.move_type == MoveType.KILL:
                p[move] = np.squeeze(action_logits[1])
                kill_cell = move.target_point
                p[move] = p[move] * np.squeeze(kill_logits)[self.CoordsToIndex(kill_cell.x, kill_cell.y)]
            else:
                p[move] = np.squeeze(action_logits[2])
        return p

    def GetV(self, net_probs):
        _, _, _, _, v = net_probs
        return np.squeeze(v)    # assuming batch size of 1
    
#     def GetResult(self, playerjm):
    def GetResult(self, player=None):
        """ Get the final game result from the viewpoint of player or current_player, if None. 
        """
        if player is None:
            player = self.current_player
        cell_map = self.field.get_cell_mapping()
        my_cells = cell_map.get(str(player), [])
        opp_cells = cell_map.get(str(1 - player), [])
        if (len(my_cells) > 0) and (len(opp_cells) <= 0):
            return 1.0
        elif (len(my_cells) <= 0) and (len(opp_cells) > 0):
            return 0.0
        else:
            return 0.5

#         jmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == playerjm])
#         notjmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 3 - playerjm])
#         if jmcount > notjmcount: return 1.0
#         elif notjmcount > jmcount: return 0.0
#         else: return 0.5 # draw

    def __repr__(self):
        s= ""
        for y in range(self.size-1,-1,-1):
            for x in range(self.size):
                s += ".01"[self.board[x][y]]
            s += "\n"
        return s


class OthelloState(GameState):
    pass
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the 
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move). 
    """
    def __init__(self,sz = 8):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [] # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0 # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)
        self.board[sz/2][sz/2] = self.board[sz/2-1][sz/2-1] = 1
        self.board[sz/2][sz/2-1] = self.board[sz/2-1][sz/2] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x,y)=(move[0],move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(x,y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x,y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a,b) in m:
            self.board[a][b] = self.playerJustMoved
    
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0 and self.ExistsSandwichedCounter(x,y)]

    def AdjacentToEnemy(self,x,y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx,dy) in [(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1),(-1,0),(-1,+1)]:
            if self.IsOnBoard(x+dx,y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                return True
        return False
    
    def AdjacentEnemyDirections(self,x,y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx,dy) in [(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1),(-1,0),(-1,+1)]:
            if self.IsOnBoard(x+dx,y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                es.append((dx,dy))
        return es
    
    def ExistsSandwichedCounter(self,x,y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx,dy) in self.AdjacentEnemyDirections(x,y):
            if len(self.SandwichedCounters(x,y,dx,dy)) > 0:
                return True
        return False
    
    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx,dy) in self.AdjacentEnemyDirections(x,y):
            sandwiched.extend(self.SandwichedCounters(x,y,dx,dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x,y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x,y))
            x += dx
            y += dy
        if self.IsOnBoard(x,y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return [] # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        jmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount: return 1.0
        elif notjmcount > jmcount: return 0.0
        else: return 0.5 # draw

    def __repr__(self):
        s= ""
        for y in range(self.size-1,-1,-1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s

