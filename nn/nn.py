import tensorflow as tf
import numpy as np
import random

class NN():
	def __init__(self):
		# Learning rate for the AdamOptimizer
		self.lr = 0.01
		# Discount factor for future rewards
		self.gamma = 0.99
		# Exploration factor
		self.epsilon = 0.9
		# Exploration decay rate
		self.ep_decay = 0.9
		# Width and height of the game board
		self.board_dim = 19
		# Dimension of the state observations
		# First dimension is batch size
		# Second dimension is a grid of 1s for every living cell belonging to player 1
		# Third dimension is a grid of 1s for every living cell belonging to player 2
		# Fourth dimension is a grid of all 0s if it's player 1's turn, and all 1s if it's player 2's turn
		self.state_dim = (None, self.board_dim, self.board_dim, 3)
		# Number of actions: birth (0), death (1), or pass (2)
		self.n_actions = 3
		# Number of epochs to train
		self.epochs = 20
		# Number of steps in each training batch (i.e. number of steps to take before updating Q)
		self.batch_size = 1000
		# Max episode length (must be less than self.batch_size) - set to 100 (max length of a game of GOLAD)
		self.max_ep_length = 100
		# Frequency to update target Q-network
		self.target_update_freq = 2
		# Tensorflow session for training
		self.sess = tf.Session()

	def setup(self):
		add_placeholders()

		# Compute Q values and grid probabilities for current state
		self.q, self.birth_logits, self.sacrifice_logits, self.death_logits = get_q_values_op('Q_scope')

		# Compute Q values of next state
		self.target_q, self.target_birth_logits, self.target_sacrifice_logits, self.target_death_logits = get_q_values_op('target_scope')

		add_loss(self.q, self.target_q)

		add_train_op('Q_scope')

		# Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Synchronize q and target_q networks
        self.sess.run(self.update_target_op)

	def add_placeholders(self):
		# State features for each time step in batch
		self.state_placeholder = tf.placeholder(tf.int32, self.state_dim)
		# True action taken at each state in the batch
		# First dimension is batch dimension
		# Second dimension is length 5: (action, birth_coord, sacrifice_coord_1, sacrifice_coord_2, death_coord)
		# Coord values are -1 if the corresponding action was not taken (i.e. if the kill action was taken, only death_coord would be >= 0)
		self.action_placeholder = tf.placeholder(tf.int32, (None, 5))
		# Reward at current time step for each state in the batch
		self.rewards = tf.placeholder(tf.float32, (None,))
		# True for each final state in the batch
		self.done_mask = tf.placeholder(tf.bool, (None,))

	def get_q_values_op(self, scope='Q_scope'):
		# Right now this is just a single conv layer + max pool + fully connected layer for each output.
		# We can add convolutional layers and change up the network architecture once this is working.

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

		# Outputs the predicted Q-values for each action
		action_Qs = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.n_actions,
			activation_fn=None,
			scope=scope+'/actions')
		# Outputs the "best" cell for a birth move
		birth_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_dim**2,
			activation_fn=None,
			scope=scope+'/birth')
		# Outputs the "best" two cells for a sacrifice move
		sacrifice_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_dim**2,
			activation_fn=None,
			scope=scope+'/sacrifice')
		# Outputs the "best" cell for a death move
		death_logits = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(pool1),
			self.board_dim**2,
			activation_fn=None,
			scope=scope+'/death')

		return action_Qs, birth_logits, sacrifice_logits, death_logits

	def add_loss(self, q, target_q):
		# Use (Q_samp(s) - Q(s, a))^2 for the Q-value loss (like in Assignment 2)
		done_mask = tf.cast(tf.logical_not(self.done_mask), tf.float32)
        Q_samp = self.rewards + done_mask * self.config.gamma * tf.reduce_max(target_q, axis=1)
        actions = tf.one_hot(self.action_placeholder[:,0], self.n_actions)
        loss = tf.square(Q_samp - tf.diag_part(tf.matmul(actions, tf.transpose(q))))
        average_loss = tf.reduce_mean(loss)
        self.loss = average_loss

        # If "birth" action taken, add squared loss for birth and sacrifice coordinate Q-values
        birth_mask = self.action_placeholder[:,0] == 0

        Q_samp = done_mask * self.config.gamma * tf.reduce_max(self.target_birth_logits, axis=1)
        birth_coords = tf.one_hot(self.action_placeholder[:,1], self.board_dim**2)
        loss = tf.square(Q_samp - tf.diag_part(tf.matmul(birth_coords, tf.transpose(self.birth_logits))))

        Q_samp = done_mask * self.config.gamma * tf.reduce_max(self.target_sacrifice_logits, axis=1)
        sacrifice_coord1 = tf.one_hot(self.action_placeholder[:,2], self.board_dim**2)
        loss += tf.square(Q_samp - tf.diag_part(tf.matmul(sacrifice_coord1, tf.transpose(self.sacrifice_logits))))

        Q_samp = done_mask * self.config.gamma * tf.reduce_max(self.target_sacrifice_logits, axis=1)
        sacrifice_coord2 = tf.one_hot(self.action_placeholder[:,3], self.board_dim**2)
        loss += tf.square(Q_samp - tf.diag_part(tf.matmul(sacrifice_coord1, tf.transpose(self.sacrifice_logits))))

        average_loss = tf.reduce_sum(birth_mask * loss) / float(tf.count_nonzero(birth_mask * loss))

        # Add birth loss to overall loss
        self.loss += average_loss

        # If "kill" action taken, add squared loss for death coordinate Q-value
        kill_mask = self.action_placeholder[:,0] == 1

        Q_samp = done_mask * self.config.gamma * tf.reduce_max(self.target_death_logits, axis=1)
        death_coords = tf.one_hot(self.action_placeholder[:,4], self.board_dim**2)
        loss = tf.square(Q_samp - tf.diag_part(tf.matmul(death_coords, tf.transpose(self.death_logits))))

        average_loss = tf.reduce_sum(kill_mask * loss) / float(tf.count_nonzero(kill_mask * loss))

        # Add death loss to overall loss
        self.loss += average_loss

	def add_train_op(self, scope='Q_scope'):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		self.train_op = optimizer.minimize(self.loss, var_list=var_list)

	def add_update_target_op(self, q_scope='Q_scope', target_scope='target_scope'):
		# Transfer Q-values learned from the network to the target Q (like in Assignment 2)
		opAssigns = []
        q_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)
        target_q_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_scope)
        for i in range(len(q_values)):
            opAssigns.append(tf.assign(target_q_values[i], q_values[i]))
        self.update_target_op = tf.group(*opAssigns)

	def train(self):
		for epoch in range(self.epochs):
			observations, actions, rewards, done_mask = self.sample()

			loss, _ = self.sess.run((self.loss, self.train_op), feed_dict={
				self.state_placeholder: observations,
				self.action_placeholder: actions,
				self.rewards: rewards,
				self.done_mask: done_mask
				})
			print "Loss for epoch %d: %.3f" % (epoch+1, loss)

			# Decay exploration factor
			self.epsilon = self.epsilon * self.ep_decay

			# Update target Q-network
			if (epoch + 1) % self.target_update_freq == 0:
				self.sess.run(self.update_target_op)

	def get_action(self, state):
		if random.random() > self.epsilon:
			# do random action and choose random coordinates
			return random.choice(range(self.n_actions), 1), random.choice(range(self.board_dim**2), 4)
		else:
			# do greedy action
			action_Qs, birth_logits, sacrifice_logits, death_logits = self.sess.run(
				(self.action_Qs, self.birth_logits, self.sacrifice_logits, self.death_logits),
				feed_dict={
				self.state_placeholder: np.expand_dims(state, 0)})
			action = np.argmax(np.squeeze(action_Qs))
			birth_coord = np.argmax(np.squeeze(birth_logits))
			# Get top two elements from sacrifice_logits
			sacrifice_coords = np.argpartition(np.squeeze(sacrifice_logits), -2)
			death_coord = np.argmax(np.squeeze(death_logits))
			return action, [birth_coord, sacrifice_coords[0], sacrifice_coords[1], death_coord]

	def sample(self):
		# Record observations from self.batch_size time steps to serve as training data
		# TODO: this needs to interface with the GOLAD engine to play an actual game
		# Should use self.get_action(state) here
		observations = []
		actions = []
		rewards = []
		done_mask = []

		return observations, actions, rewards, done_mask

if __name__ == '__main__':
	model = NN()
	model.setup()
	model.train()