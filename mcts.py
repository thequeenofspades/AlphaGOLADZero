
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
import random

from field.field import Field
from game_state import GameState
from game_state import GOLADState
from move.move_type import MoveType

from nn.nn import NN

from config import config


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
        self.player = state.current_player # 0(me) or 1(opponent) TODO: might not be necessary since perspective of v is taken care of by NN
        self.state = state
        
    def UCTSelectChild(self, c_puct=1.0):
        """ PUCT algorithm
        """
        node = sorted(self.childNodes, key = lambda c: c.total_rewards/(c.total_visits+1e-10) + c_puct * c.prior * np.sqrt(self.total_visits)/(1 + c.total_visits))[-1]
        return node
    
    def AddChild(self, m, s, p_m):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(player=1-self.player, move=m, parent=self, state=s, prior=p_m)
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

def extract_p_move(p, m, all_ms, nn):
    if len(p.shape) > 1:
        p = np.squeeze(p)   # remove batch dimension
    if m.move_type == MoveType.PASS:
        return p[-1]
    elif m.move_type == MoveType.KILL:
        return p[nn.coords_to_idx(m.target_point.x, m.target_point.y)]
    elif m.move_type == MoveType.BIRTH:
        N_birth_moves = np.sum([(_m.move_type==MoveType.BIRTH) and (_m.target_point==m.target_point) for _m in all_ms]) # number of birth moves at target point of given move
        assert(N_birth_moves > 0)
        return p[nn.coords_to_idx(m.target_point.x, m.target_point.y)] / N_birth_moves
    else:
        assert False
    

def UCT(rootstate, itermax, nn, verbose = False, rootnode = None):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 0 starts), with game rewards {-1, +1}."""

    if rootnode is None:
        rootnode = Node(player=0, state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        v = 0

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand and Evaluate - use NN to evalute leaf node
        if node.untriedMoves != []:
            # p, v = state.GetP(), state.GetV() # get outputs from NN
            p, v = nn.evaluate(state.Convert()) # get outputs from NN
            all_ms = list(node.untriedMoves) # create copy 
            
            # add Dirichlet noise if rootnode
            if node == rootnode:
                eps = 0.25
            else:
                eps = 0.
            p_moves = np.array([(1 - eps) * extract_p_move(p, m, all_ms, nn) for m in all_ms]) + eps * np.random.dirichlet([0.03]*len(all_ms))
            assert np.amin(p_moves) >= 0

            # beam search
            beam_width = min(10, len(p_moves))
            idxs = np.argsort(p_moves)[-beam_width:] # index of moves with highest prob
            for idx in idxs:
                m = all_ms[idx]
                temp_state = state.Clone()
                temp_state.DoMove(m)
                # compute p_move from p
                # p_move = extract_p_move(p, m, all_ms, nn)
                p_move = p_moves[idx]
                node.AddChild(m, temp_state, p_move) 

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            if node.player == state.current_player:
                node.Update(v) # Update node with result from POV of node.player
            else:
                node.Update(-v)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    # if (verbose): print rootnode.TreeToString(0)
    # else: print rootnode.ChildrenToString()

    # Select move to play using exponentiated visit count
    tau = 1.
    exp_visits = np.array([np.power(c.total_visits, 1./tau) for c in rootnode.childNodes]) + 1e-10
    pi = exp_visits / np.sum(exp_visits)

    pi_t = np.zeros((nn.board_w * nn.board_h + 1))
    move_tuples = [(c.move.move_type, c.move.target_point, c.move.sacrifice_points) for c in rootnode.childNodes]
    for i, move_tuple in enumerate(move_tuples):
        if move_tuple[1] is not None:
            pi_t[nn.coords_to_idx(move_tuple[1].x, move_tuple[1].y)] = pi[i] # TODO: check dtype of move.target_point
        else: # pass
            assert move_tuple[0] == MoveType.PASS
            pi_t[-1] = pi[i]
    
    return np.random.choice(rootnode.childNodes, p=pi), pi_t # return child node sampled from pi and pi_t

def init_cells(width = 18, height = 16, cells_each_player = 50):
    assert width & 1 == 0
    assert height & 1 == 0
    assert (cells_each_player * 2) < (width * height)
    cells = ["." for _ in xrange(width * height)]
    for idx in random.sample(range(width * height / 2), cells_each_player):
        cells[idx] = "0"
        cells[width * height - idx - 1] = "1"
    cells_str = ''.join((cell + ",") for cell in cells)[:-1]
    return cells_str
    
def UCTPlayGame(nn, nn2=None):
    """ Self-play using MCTS, returns s_t's, pi_t's, and z to use for training.
    """
    width = config.board_width
    height = config.board_height
    cells_each_player = config.cells_each_player

    field = Field()
    field.width = width
    field.height = height
    field.parse(init_cells(field.width, field.width, cells_each_player))
    state = GOLADState(field)

    data = {}
    data['s'] = []
    data['pi'] = []
    c = None
    current_nn = nn # use nn for first player
    while (state.GetMoves() != []):
        c, pi= UCT(rootstate = state, itermax = config.mcts_itermax, nn=current_nn, verbose = False, rootnode = c)
        m = c.move
        data['s'].append(state.Convert())
        data['pi'].append(pi)
        if config.verbose:
            print ("\nTurn {}, Player {}, Best Move: {}" \
                .format(state.timestep, state.current_player, str(m)))
            state.field.pprint()
        state.DoMove(m)
        if nn2 is not None:
            if current_nn == nn:
                current_nn = nn2
            else:
                current_nn = nn

    if config.verbose:
        print('Result: {}'.format(state.GetResult(0)))
    data['z'] = [[state.GetResult(0)]] * len(data['s']) # get result from perspective of first player (ie rootnode)
    
    return data
#     
#     bot = Bot()
#     game = Game()
#     game.run(bot)
    
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
    nn = NN(config)
    nn.setup()
    UCTPlayGame(nn)
    
