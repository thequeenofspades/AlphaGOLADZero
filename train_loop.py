import numpy as np
from nn.nn import NN
from config import config
from mcts import UCTPlayGame

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """    
    nn = NN(config)
    nn.setup()
    
    batch_data = {}
    batch_data['s'], batch_data['pi'], batch_data['z'] = ([], [], [])

    for _ in range(config.n_iters):
        
        # Collect self-play data from MCTS
        while len(batch_data['s']) < config.buffer_size:
            if config.verbose:
                print('Current data size: {}'.format(len(batch_data['s'])))
            data = UCTPlayGame(nn)
            for k in batch_data.keys():
                batch_data[k].extend(data[k])
        
        nn.train((batch_data['s'], batch_data['pi'], batch_data['z']))
        
        for k in batch_data.keys():
            batch_data[k] = batch_data[k][config.batch_size:]