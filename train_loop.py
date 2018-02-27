import numpy as np
from nn.nn import NN
from config import config
from mcts import UCTPlayGame

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """    
    nn = NN(config)
    nn.setup()
    
    n_iters = 10
    for _ in range(n_iters):
        batch_data = {}
        batch_data['s'], batch_data['pi'], batch_data['z'] = ([], [], [])
        while len(batch_data['s']) < config.batch_size:
            if config.verbose:
                print('Current data size: {}'.format(len(batch_data['s'])))
            data = UCTPlayGame(nn)
            for k in batch_data.keys():
                batch_data[k].extend(data[k])
        
        batch_sample = (batch_data['s'][:config.batch_size], batch_data['pi'][:config.batch_size], batch_data['z'][:config.batch_size])
        nn.train(batch_sample)