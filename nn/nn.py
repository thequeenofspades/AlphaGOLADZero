import tensorflow as tf
import numpy as np
import random

class NN():
    def __init__(self, config):
        self.config = config
        # Learning rate for the AdamOptimizer
        self.lr = config.lr
        # Width of the game board
        self.board_w = config.board_width
        # Height of the game board
        self.board_h = config.board_height
        # Number of actions: birth (0), death (1), or pass (2)
        self.n_actions = config.n_actions
        # Number of steps to train
        self.train_steps = config.train_steps
        # Number of examples in each training batch (i.e. number of steps to take before updating network)
        self.batch_size = config.batch_size
        # Max episode length (must be less than self.batch_size) - set to 100 (max length of a game of GOLAD)
        self.max_ep_length = config.max_ep_length
        # Tensorflow session for training
        self.sess = tf.Session()
        # Directory to save/restore trained weights
        self.save_path = config.save_path
        # How often to save weights
        self.save_freq = config.save_freq
        # Internal count of train steps
        self._steps = 0
        # How many residual blocks in the residual tower
        self.res_tower_height = config.res_tower_height

    def setup(self):
        self.add_placeholders()

        # Compute Q values and grid probabilities for current state
        self.alpha_go_zero_network('scope')

        # Minimize difference between network and MCTS outputs
        self.add_loss()

        self.add_train_op('scope')

        # Initialize all variables or restore from saved checkpoint
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print "Restored weights from checkpoint"
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save_weights(self):
        self.saver.save(self.sess, self.save_path + 'model_step_{}.ckpt'.format(self._steps))

    def add_placeholders(self):
        # State features for each time step in batch
        self.state_placeholder = tf.placeholder(tf.float32, (None, self.board_w, self.board_h, 3))
        # Self-play winner z
        self.z = tf.placeholder(tf.float32, (None, 1))
        # Action probability distribution over grid output by MCTS - kill/birth at each location or pass
        self.mcts_probs = tf.placeholder(tf.float32, (None, self.board_w*self.board_h + 1))
        # Records whether we are training or evaluating
        self.training_placeholder = tf.placeholder(tf.bool, ())

    def conv_block(self, X, is_training, scope='scope'):
        conv = tf.contrib.layers.conv2d(
            X,
            256,
            3,
            stride=1,
            scope=scope+'/conv')
        norm = tf.contrib.layers.batch_norm(
            conv,
            is_training=is_training,
            scope=scope+'/norm')
        relu = tf.nn.relu(norm)
        return relu

    def residual_block(self, X, index, is_training, scope='scope'):
        conv1 = tf.contrib.layers.conv2d(
            X,
            256,
            3,
            stride=1,
            scope=scope+'/res_conv1_'+str(index))
        norm1 = tf.contrib.layers.batch_norm(
            conv1,
            is_training=is_training,
            scope=scope+'/res_norm1_'+str(index))
        relu1 = tf.nn.relu(norm1)
        conv2 = tf.contrib.layers.conv2d(
            relu1,
            256,
            3,
            stride=1,
            scope=scope+'/res_conv2_'+str(index))
        norm2 = tf.contrib.layers.batch_norm(
            conv2,
            is_training=is_training,
            scope=scope+'/res_norm2_'+str(index))
        skip = conv1 + X
        relu2 = tf.nn.relu(skip)
        return relu2

    def policy_head(self, X, is_training, scope='scope'):
        conv = tf.contrib.layers.conv2d(
            X,
            2,
            1,
            stride=1,
            scope=scope+'/policy_head_conv')
        norm = tf.contrib.layers.batch_norm(
            conv,
            is_training=is_training,
            scope=scope+'/policy_head_norm')
        relu = tf.nn.relu(norm)
        output = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(relu),
            self.board_w * self.board_h + 1,
            activation_fn=tf.nn.softmax,
            scope=scope+'/policy_head_output')
        return output

    def value_head(self, X, is_training, scope='scope'):
        conv = tf.contrib.layers.conv2d(
            X,
            1,
            1,
            stride=1,
            scope=scope+'/value_head_conv')
        norm = tf.contrib.layers.batch_norm(
            conv,
            is_training=is_training,
            scope=scope+'/value_head_norm')
        relu = tf.nn.relu(norm)
        hidden = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(relu),
            256,
            activation_fn=tf.nn.relu,
            scope=scope+'/value_head_hidden')
        output = tf.contrib.layers.fully_connected(
            hidden,
            1,
            activation_fn=tf.nn.tanh,
            scope=scope+'/value_head/output')
        return output

    def alpha_go_zero_network(self, scope='scope'):
        # Duplicates the architecture from the AlphaGoZero nature paper
        convolutional_block = self.conv_block(self.state_placeholder, self.training_placeholder, scope)
        res_tower = [convolutional_block]
        for i in range(1, self.res_tower_height):
            res_block = self.residual_block(res_tower[i - 1], i, self.training_placeholder, scope)
            res_tower.append(res_block)
        self.probs = self.policy_head(res_tower[-1], self.training_placeholder, scope)
        self.v = self.value_head(res_tower[-1], self.training_placeholder, scope)

    def add_loss(self):
        # Minimize error between network predictions and MCTS predictions
        self.loss = tf.square(self.z - self.v)
	self.loss = self.loss - tf.reduce_sum(tf.multiply(self.mcts_probs, tf.log(self.probs)), 1)
        #self.loss = self.loss - tf.losses.log_loss(labels=self.mcts_probs, predictions=self.probs)
        self.loss = tf.reduce_mean(self.loss)

    def add_train_op(self, scope='Q_scope'):
        # Minimize training loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def evaluate(self, states):
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=0) # add batch dimension
        probs, v = self.sess.run((self.probs, self.v), feed_dict={
            self.state_placeholder: states,
            self.training_placeholder: False
            })
        return probs, v

    def train(self, data):
        states, mcts_probs, z = (np.array(x) for x in data)
        assert len(states) == len(mcts_probs) == len(z)
        avg_loss = 0.0
        for step in range(self.train_steps):
            idx = np.random.choice(range(len(states)), self.batch_size, replace=False)
            loss, _ = self.sess.run((self.loss, self.train_op), feed_dict={
                self.state_placeholder: states[idx],
                self.z: z[idx],
                self.mcts_probs: mcts_probs[idx],
                self.training_placeholder: True
                })
            avg_loss += loss
            if (step + 1) % self.save_freq == 0:
                print "Saved weights after %d steps" % (step+1)
                self.save_weights()
            self._steps += 1
        print "Average loss: %f" % (avg_loss / float(self.train_steps))
            
    def coords_to_idx(self, x, y, major='col'):
        if major == 'col':
            return x * self.board_h + y
        elif major == 'row':
            return x * self.board_w + y
        else:
            assert False, "major must be 'row' or 'col'"
