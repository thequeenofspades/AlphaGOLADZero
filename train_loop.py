import numpy as np
from nn.nn import NN
from config import config
from mcts import UCTPlayGame
import cPickle as pickle
import os

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """    
    nn = NN(config)
    nn.setup()
    
    batch_data = {}
    batch_data['s'], batch_data['pi'], batch_data['z'] = ([], [], [])
    data_path = os.path.join(config.save_path, 'batch_data_iter_{}.pkl')

    for iteration in range(config.n_iters):
        
        # Collect self-play data from MCTS
        while len(batch_data['s']) < config.buffer_size:
            if config.verbose:
                print('Current data size: {}'.format(len(batch_data['s'])))
            data = UCTPlayGame(nn)
            for k in batch_data.keys():
                batch_data[k].extend(data[k])

        print "Finished collecting self-play data from iteration %d" % iteration
        
        nn.train((batch_data['s'], batch_data['pi'], batch_data['z']))
        
        print "Saving batch data..."
        fn = data_path.format(iteration)
        pickle.dump(batch_data, open(fn, 'wb'), pickle.HIGHEST_PROTOCOL)
        print "Done saving {}".format(fn)

        print "Finished training on self-play data from iteration %d" % iteration
        
        for k in batch_data.keys(): # remove oldest 25% from buffer
            batch_data[k] = batch_data[k][int(0.25*config.buffer_size):]