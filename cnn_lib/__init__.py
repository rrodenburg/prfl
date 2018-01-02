import tensorflow as tf
import numpy as np

class Network(object):

	def __init__(
				self,
				width = 84,
				heigth = 84,
				frame_stacks = 4,
				learning_rate = 0.00025,
				mini_batch_size = 16,
				network_name = 'nature_cnn'
				):

		network_name_dict = {'nature_cnn' : self.nature_cnn}

		tf.reset_default_graph()

		self.width = width
		self.heigth = heigth
		self.frame_stacks = frame_stacks
		self.learning_rate = learning_rate
		self.mini_batch_size = mini_batch_size
		self.network = network_name_dict[network_name]

		self.network_param_init()
		self.output, self.best_action, self.max_Q_value = self.network()
		self.loss = self.loss_func()
		self.train_op = self.train()

	def network_param_init(self):

		self.x = tf.placeholder("float32",(None, self.width, self.heigth, self.frame_stacks)) #input states
		self.y = tf.placeholder("float32",(None)) #y-values for loss function, as described in atari paper
		self.a = tf.placeholder("int32",(None)) #actions played in batch; 0: nothing, 1: up, 2: down
	
		#return x, y, a

	def tf_session_init(self):
		
		#tf.reset_default_graph()

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

	def nature_cnn(self):

		# Convolutional Layers
		conv1 = tf.layers.conv2d(
		  inputs = self.x,
		  filters = 32,
		  kernel_size = [8, 8],
		  strides = (4,4),
		  padding = "valid", #valid means no padding
		  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
		  bias_initializer=tf.zeros_initializer(),
		  activation = tf.nn.relu) #output is 20x20x32

		conv2 = tf.layers.conv2d(
		  inputs = conv1,
		  filters = 64,
		  kernel_size = [4, 4],
		  strides = (2,2),
		  padding = "valid", #valid means no padding
		  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
		  bias_initializer=tf.zeros_initializer(),
		  activation = tf.nn.relu) #output is 9x9x64

		conv3 = tf.layers.conv2d(
		  inputs = conv2,
		  filters = 64,
		  kernel_size = [3, 3],
		  strides = (1,1),
		  padding = "valid", #valid means no padding
		  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
		  bias_initializer=tf.zeros_initializer(),
		  activation = tf.nn.relu) #output is 7x7x64

		# Output layer (dense layer)
		conv3_flat = tf.reshape(conv3,[-1,7*7*64])
		FC = tf.layers.dense(inputs=conv3_flat, units = 512, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu, bias_initializer=tf.zeros_initializer())
		output = tf.layers.dense(inputs = FC, units = 3,  kernel_initializer = tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
		best_action = tf.argmax(input = output, axis = 1)
		max_Q_value = tf.reduce_max(output, axis = 1, name = 'Q_max')

		return output, best_action, max_Q_value

	def loss_func(self):

		onehot_actions = tf.one_hot(indices = self.a, depth=3) #batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
		onehot_actions = tf.transpose(onehot_actions)
		Q_values = tf.diag_part(tf.matmul(self.output, onehot_actions))
		loss = tf.reduce_mean(tf.square(self.y - Q_values), axis = 0)
	
		return loss

	def train(self):
		optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, momentum = 0.95, epsilon = 0.01)
	
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(
	            loss = self.loss,
	            global_step=global_step)

		return train_op

	def mini_batch_sample(self, memory, mini_batch_size):
		#Select random mini-batch of 32 transitions
		idx = np.random.randint(len(memory), size = mini_batch_size)
		mini_batch = [memory[i] for i in idx]
	
		mini_batch_x = [x[0].astype('float32') for x in mini_batch]
		mini_batch_y = [x[3].astype('float32') for x in mini_batch]
		reward = [x[2] for x in mini_batch]
	
		mini_batch_action = [x[1] for x in mini_batch]
		return mini_batch_x, mini_batch_y, mini_batch_action, reward

	def create_y(self, reward, q_values):
		return [q_values[idx] if x is 0 else reward[idx] for idx,x in enumerate(reward)]
	
	def pick_greedy_action(self, frame_stack):
		frame_stack = [[x.astype('float32') for x in frame_stack]]
		action, q_value = self.sess.run([self.best_action, self.max_Q_value], feed_dict = {self.x : frame_stack})
		return action[0], q_value[0]
	
	def backprop(self, memory, mini_batch_size):
		mini_batch_x, mini_batch_y, mini_batch_action, reward = self.mini_batch_sample(memory, mini_batch_size)
		_, q_value = self.sess.run([self.best_action, self.max_Q_value], {self.x: mini_batch_y})
	
		target_y = self.create_y(reward, q_value) # select reward as y if episode had ended
	
		feed_dict_train = {
						self.x : mini_batch_x,
						self.y : target_y,
						self.a : mini_batch_action
						}
	
		_, loss_value = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict_train)
	
		return loss_value

	def update_nn(self, dataset, mini_batch_size, loss_value, backprop_cycles, total_transition_count, replay_start_size):
		if total_transition_count > replay_start_size:
			loss_value += self.backprop(dataset, mini_batch_size)
			backprop_cycles += 1
		return loss_value, backprop_cycles