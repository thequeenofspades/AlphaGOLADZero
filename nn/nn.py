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
		add_placeholders()

		# Compute Q values and grid probabilities for current state
		get_q_values_op('Q_scope')

		# Minimize difference between network and MCTS outputs
		add_loss()

		add_train_op('Q_scope')

		# Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

	def add_placeholders(self):
		# State features for each time step in batch
		self.state_placeholder = tf.placeholder(tf.int32, (None, self.board_w, self.board_h, 3))
		# Self-play winner z
		self.z = tf.placeholder(tf.int32, (None,))
		# Action probabilities output by MCTS
		self.action_probs = tf.placeholder(tf.float32, (None, 3))
		# Birth coordinate probabilities output by MCTS
		self.birth_probs = tf.placeholder(tf.float32, (None, self.board_w*self.board_h))
		# Sacrifice coordinate probabilities output by MCTS
		self.sac_probs = tf.placeholder(tf.float32, (None, self.board_w*self.board_h))
		# Kill coordinate probabilities output by MCTS
		self.kill_probs = tf.placeholder(tf.float32, (None, self.board_w*self.board_h))

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

		# Outputs the predicted probabilities for each action
		self.action_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.n_actions,
			activation_fn=tf.nn.sigmoid,
			scope=scope+'/actions')
		# Outputs the predicted probabilities for "birthing" in each cell
		Outputs the "best" cell for a birth move
		self.birth_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_w*self.board_h,
			activation_fn=tf.nn.sigmoid,
			scope=scope+'/birth')
		# Outputs the "best" two cells for a sacrifice move
		self.sac_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_w*self.board_h,
			activation_fn=tf.nn.sigmoid,
			scope=scope+'/sacrifice')
		# Outputs the "best" cell for a death move
		self.kill_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_w*self.board_h,
			activation_fn=tf.nn.sigmoid,
			scope=scope+'/death')
		# Outputs the predicted winner v
		self.v = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			1,
			activation_fn=tf.nn.tanh,
			scope=scope+'/v')

	def add_loss(self):
		# Minimize error between network predictions and MCTS predictions
		self.loss = tf.square(self.z - self.v)
		self.loss = self.loss - tf.losses.log_loss(labels=self.action_probs, predictions=self.action_logits)
		self.loss = self.loss - tf.losses.log_loss(labels=self.birth_probs, predictions=self.birth_logits)
		self.loss = self.loss - tf.losses.log_loss(labels=self.sac_probs, predictions=self.sac_logits)
		self.loss = self.loss - tf.losses.log_loss(labels=self.kill_probs, predictions=self.kill_logits)

	def add_train_op(self, scope='Q_scope'):
		# Minimize training loss
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op = optimizer.minimize(self.loss)

	def predict(self, states):
		output_logits = (self.action_logits, self.birth_logits, self.sac_logits, self.kill_logits, self.v)
		output_logits = self.sess.run(output_logits, feed_dict={self.state_placeholder: states})
		return output_logits

	def update(self, batch_sample):
		states, z, action_probs, birth_probs, sac_probs, kill_probs = batch_sample
		for epoch in range(self.epochs):
			loss, _ = self.sess.run((self.loss, self.train_op), feed_dict={
				self.state_placeholder: states,
				self.z: z,
				self.action_probs: action_probs,
				self.birth_probs: birth_probs,
				self.sac_probs: sac_probs,
				self.kill_probs: kill_probs
				})
			print "Loss for epoch %d: %.3f" % (epoch+1, loss)