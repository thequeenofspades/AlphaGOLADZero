
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

from game_state import GameState
from game_state import GOLADState

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
    UCTPlayGame()
    
