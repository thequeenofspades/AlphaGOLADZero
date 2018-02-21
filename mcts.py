
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


class NimState:
    """ A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips with the 
        winner being the player to take the last chip. 
        In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for player 1
        (by choosing k) chips.
        Any initial state of the form 4n is a win for player 2.
    """
    def __init__(self, ch):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.chips = ch
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.chips)
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        assert move >= 1 and move <= 3 and move == int(move)
        self.chips -= move
        self.playerJustMoved = 3 - self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return range(1,min([4, self.chips + 1]))
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        assert self.chips == 0
        if self.playerJustMoved == playerjm:
            return 1.0 # playerjm took the last chip and has won
        else:
            return 0.0 # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Chips:" + str(self.chips) + " JustPlayed:" + str(self.playerJustMoved)
        return s

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s

class OthelloState:
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
    state = GOLADState()
    while (state.GetMoves() != []):
        print str(state)
        m = UCT(rootstate = state, itermax = 1000, verbose = False) 
        print "Best Move: " + str(m) + "\n"
        state.DoMove(m)
    if state.GetResult() == 1.0:
        print "Player " + str(state.get_player()) + " wins!"
    elif state.GetResult() == 0.0:
        print "Player " + str(1-state.get_player()) + " wins!"
    else: print "Nobody wins!"

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()
    
    
    
import random
import math
import hashlib
import logging
import argparse


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class State():
	NUM_TURNS = 10	
	GOAL = 0
	MOVES=[2,-2,3,-3]
	MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
	num_moves=len(MOVES)
	def __init__(self, value=0, moves=[], turn=NUM_TURNS):
		self.value=value
		self.turn=turn
		self.moves=moves
	def next_state(self):
		nextmove=random.choice([x*self.turn for x  in self.MOVES])
		next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)
		return next
	def terminal(self):
		if self.turn == 0:
			return True
		return False
	def reward(self):
		r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
		return r
	def __hash__(self):
		return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	def __repr__(self):
		s="Value: %d; Moves: %s"%(self.value,self.moves)
		return s
	

class Node():
	def __init__(self, state, parent=None):
		self.visits=1
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)
	def update(self,reward):
		self.reward+=reward
		self.visits+=1
	def fully_expanded(self):
		if len(self.children)==self.state.num_moves:
			return True
		return False
	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s
		


def UCTSEARCH(budget,root):
	for iter in range(int(budget)):
		if iter%10000==9999:
			logger.info("simulation: %d"%iter)
			logger.info(root)
		front=TREEPOLICY(root)
		reward=DEFAULTPOLICY(front.state)
		BACKUP(front,reward)
	return BESTCHILD(root,0)

def TREEPOLICY(node):
	#a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
	while node.state.terminal()==False:
		if len(node.children)==0:
			return EXPAND(node)
		elif random.uniform(0,1)<.5:
			node=BESTCHILD(node,SCALAR)
		else:
			if node.fully_expanded()==False:	
				return EXPAND(node)
			else:
				node=BESTCHILD(node,SCALAR)
	return node

def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children:
		new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))	
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		logger.warn("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='MCTS research code')
	parser.add_argument('--num_sims', action="store", required=True, type=int)
	parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS))
	args=parser.parse_args()
	
	current_node=Node(State())
	for l in range(args.levels):
		current_node=UCTSEARCH(args.num_sims/(l+1),current_node)
		print("level %d"%l)
		print("Num Children: %d"%len(current_node.children))
		for i,c in enumerate(current_node.children):
			print(i,c)
		print("Best Child: %s"%current_node.state)
		
print("--------------------------------") 
