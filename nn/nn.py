import tensorflow as tf
import numpy as np
import random

class NN():
    def __init__(self):
        # Learning rate for the AdamOptimizer
        self.lr = 0.01
        # Width of the game board
        self.board_w = 18
        # Height of the game board
        self.board_h = 16
        # Number of actions: birth (0), death (1), or pass (2)
        self.n_actions = 3
        # Number of epochs to train
        self.epochs = 20
        # Number of steps in each training batch (i.e. number of steps to take before updating network)
        self.batch_size = 1000
        # Max episode length (must be less than self.batch_size) - set to 100 (max length of a game of GOLAD)
        self.max_ep_length = 100
        # Tensorflow session for training
        self.sess = tf.Session()

    def setup(self):
        self.add_placeholders()

        # Compute Q values and grid probabilities for current state
        self.get_q_values_op('Q_scope')

        # Minimize difference between network and MCTS outputs
        self.add_loss()

        self.add_train_op('Q_scope')

        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholders(self):
        # State features for each time step in batch
        self.state_placeholder = tf.placeholder(tf.float32, (None, self.board_w, self.board_h, 3))
        # Self-play winner z
        self.z = tf.placeholder(tf.float32, (None,))
        # Action probability distribution over grid output by MCTS - kill/birth at each location or pass
        self.mcts_probs = tf.placeholder(tf.float32, (None, self.board_w*self.board_h + 1))

    def get_q_values_op(self, scope='Q_scope'):
        # Right now this is just a single conv layer + max pool + fully connected layer for each output.
        # We can add convolutional layers and change up the network architecture once this is working.
        # Also need to add L2 regularization to match AlphaGoZero nature paper

        # Single conv2d layer
        conv1 = tf.contrib.layers.conv2d(
            self.state_placeholder,
            16,
            7,
            scope=scope+'/conv1'
            )

        # Single max pooling layer
        pool1 = tf.contrib.layers.max_pool2d(
            conv1,
            2,
            scope=scope+'/pool1')

        # Outputs the move probability distribution over the grid
        self.probs = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(pool1),
            self.board_w * self.board_h + 1,
            activation_fn=tf.nn.softmax,
            scope=scope+'/probs')

        # Outputs the predicted winner v
        self.v = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(pool1),
            1,
            activation_fn=tf.nn.tanh,
            scope=scope+'/v')

    def add_loss(self):
        # Minimize error between network predictions and MCTS predictions
        self.loss = tf.square(self.z - self.v)
        self.loss = self.loss - tf.losses.log_loss(labels=self.mcts_probs, predictions=self.probs)

    def add_train_op(self, scope='Q_scope'):
        # Minimize training loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def evaluate(self, states):
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=0) # add batch dimension
        probs, v = self.sess.run((self.probs, self.v), feed_dict={self.state_placeholder: states})
        return probs, v

    def train(self, batch_sample):
        states, mcts_probs, z = batch_sample
        for epoch in range(self.epochs):
            loss, _ = self.sess.run((self.loss, self.train_op), feed_dict={
                self.state_placeholder: states,
                self.z: z,
                self.mcts_probs: mcts_probs
                })
            print "Loss for epoch %d: %.3f" % (epoch+1, loss)
            
    def coords_to_idx(self, x, y, major='col'):
        if major == 'col':
            return x * self.board_h + y
        elif major == 'row':
            return x * self.board_w + y
        else:
            assert False, "major must be 'row' or 'col'"
