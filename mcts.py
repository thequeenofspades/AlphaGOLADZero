
# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

import numpy as np

import itertools

from bot.player import Player
from bot.game import Game
from field.point import Point
from field.field import Field
from move.move import Move
from move.move_type import MoveType

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
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the 
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move). 
    """
    def __init__(self, field, myid="0", oppid="1"):
        self.playerJustMoved = 1 # At the root pretend the player just moved is p1 - p0 has the first move
        self.field = field
        self.myid = myid
        self.oppid = oppid

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GOLADState()
        st.playerJustMoved = self.playerJustMoved
        st.field = self.field.Clone()
        st.myid = self.myid
        st.oppid = self.oppid
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        # Apply cell change
        if move[0] == MoveType.KILL:
            self.field.cell[move[1].x][move[1].y] = '.'
        elif move[0] == MoveType.BIRTH:
            self.field.cell[move[1].x][move[1].y] = self.myid
            self.field.cell[move[2].x][move[2].y] = '.'
            self.field.cell[move[3].x][move[3].y] = '.'
        elif move[0] == MoveType.PASS:
            pass

        # Simulate the game for 1 step
        cell_map = self.field.get_cell_mapping()
        dead_cells = cell_map.get('.', [])
        my_cells = cell_map.get(self.myid, [])
        opp_cells = cell_map.get(self.oppid, [])
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

        # Flip turn player
        self.playerJustMoved = 1 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        moves = []
#         curr_player_cell = "0" if self.playerJustMoved==0 else "1"
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
        my_cells = cell_map.get(self.myid, [])
        opp_cells = cell_map.get(self.oppid, [])
        living_cells = my_cells + opp_cells
        # Generate kill moves
        for kill_cell in living_cells:
            moves.append(Move(MoveType.KILL, kill_cell))
        # Generate birth moves
        for birth_cell in dead_cells:
            for sacrifice_cells in itertools.combinations(my_cells, 2):
                moves.append(Move(MoveType.BIRTH, birth_cell, sacrifice_cells[0], sacrifice_cells[1]))
        # Generate pass move
        moves.append(Move(MoveType.PASS))
        return moves

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.field.width and y >= 0 and y < self.field.height
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        cell_map = self.field.get_cell_mapping()
        my_cells = cell_map.get(self.myid, [])
        opp_cells = cell_map.get(self.oppid, [])
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


class Node:
    """ A node in the game tree. 
    """
    def __init__(self, player, move=None, parent=None, state=None, prior=None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = [] # list of child nodes
        self.total_rewards = 0
        self.total_visits = 0
        self.prior = prior
        self.untriedMoves = state.GetMoves() # future child nodes
        self.player = state.get_player() # 0(me) or 1(opponent) TODO: might not be necessary since perspective of v is taken care of by NN
        self.state = state
        
    def UCTSelectChild(self, c_puct=1.0):
        """ PUCT algorithm
        """
        node = sorted(self.childNodes, key = lambda c: c.total_rewards/(c.total_visits+1e-10) + c_puct * c.prior * np.sqrt(self.total_visits)/(1 + c.total_visits))[-1]
        return node
    
    def AddChild(self, m, s, p):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(player=1-self.player, move=m, parent=self, state=s, prior=p[m])
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, v):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.total_visits += 1
        self.total_rewards += v

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.total_rewards) + "/" + str(self.total_visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 0 starts), with game rewards {-1, +1}."""

    rootnode = Node(player=0, state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand and Evaluate - use NN to evalute leaf node
        p, v = state.GetResult() # get outputs from NN
        for m in node.untriedMoves:
            temp_state = state.Clone()
            temp_state.DoMove(m)
            node.AddChild(m, temp_state, p)
      
        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            if node.player == state.get_player():
                node.Update(v) # Update node with result from POV of node.player
            else:
                node.Update(-v)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print rootnode.TreeToString(0)
    else: print rootnode.ChildrenToString()

    # Select move to play using exponentiated visit count
    tau = 1.
    exp_visits = np.array([np.pow(c.total_visits, 1./tau) for c in rootnode.childNodes])
    pi = exp_visits / np.sum(exp_visits)
    return np.random.choice(rootnode.childNodes, p=pi).move # return move sampled from pi
                
def UCTPlayGame():
    """ Self-play using MCTS, returns s_t's, pi_t's, and z to use for training.
    """
    
    bot = Bot()
    game = Game()
    game.run(bot)
    
    # TODO: check result
    
    # Original Othello impl
#     state = GOLADState()
#     while (state.GetMoves() != []):
#         print str(state)
#         m = UCT(rootstate = state, itermax = 1000, verbose = False) 
#         print "Best Move: " + str(m) + "\n"
#         state.DoMove(m)
#     if state.GetResult() == 1.0:
#         print "Player " + str(state.get_player()) + " wins!"
#     elif state.GetResult() == 0.0:
#         print "Player " + str(1-state.get_player()) + " wins!"
#     else: print "Nobody wins!"

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()
    
