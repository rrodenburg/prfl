import tensorflow as tf
import numpy as np
import os
import shutil

class Network(object):

	def __init__(
				self,
				width = 84,
				heigth = 84,
				frame_stacks = 4,
				learning_rate = 0.00025,
				mini_batch_size = 16,
				network_name = 'nature_cnn',
				logdir = './tmp/1'
				):

		network_name_dict = {'nature_cnn' : self.nature_cnn}

		tf.reset_default_graph()

		self.width = width
		self.heigth = heigth
		self.frame_stacks = frame_stacks
		self.learning_rate = learning_rate
		self.mini_batch_size = mini_batch_size
		self.network = network_name_dict[network_name]
		self.logdir = logdir

		while os.path.isdir(self.logdir) == True:
			self.logdir = self.logdir[:-1] + str(int(self.logdir[-1]) + 1)
			
			#shutil.rmtree(self.logdir)
		
		print('dir exists :', os.path.isdir(self.logdir))
		print('tensorboard data is saved in :', self.logdir)

		### Initiate training and action picking graph
		
		
		self.x_t, self.y, self.a = self.network_param_init()
		self.output, self.best_action, self.max_Q_value_t = self.network(self.x_t, '_train')
		self.loss = self.loss_func()
		self.train_op, self.global_step = self.train()

		
		#self.variables_t = tf.trainable_variables(scope = None)
		self.variables_t = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'network_train')

		self.average_score, self.n_games_played, self.epoch_time ,self.played_q_value = self.stats_var_init()
		
		self.summ_per_state = tf.summary.merge_all(key = 'per_state')
		self.summ_per_epoch = tf.summary.merge_all(key = 'per_epoch')

		#([self.loss_sum])
		#self.summ_loss = tf.summary.merge([self.loss_sum, self.image_summ])
		#self.epoch_stats = tf.summary.merge([self.avg_score_summ, self.n_games_played_summ])
		#self.epoch_stats = tf.summary.merge(['average_score'])

		### Initiate graph to generate y as backprop target

		self.x_pred, _, _ = self.network_param_init()
		_, self.best_action_pred, self.max_Q_value_pred = self.network(self.x_pred, '_predict')

		#self.init_pred = tf.global_variables_initializer()
		self.variables_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'network_predict')
		#self.summ_pred = tf.summary.merge_all()

		### Initialize variables
		self.init = tf.global_variables_initializer()

	def network_param_init(self):

		x = tf.placeholder("float32",(None, self.width, self.heigth, self.frame_stacks), name = 'x') #input states
		y = tf.placeholder("float32",(None), name = 'y') #y-values for loss function, as described in atari paper
		a = tf.placeholder("int32",(None), name = 'actions') #actions played in batch; 0: nothing, 1: up, 2: down

		tf.summary.image('input', x, 3, collections = ['per_state'])
	
		return x, y, a

	def stats_var_init(self):
		with tf.name_scope("epoch_stats"):
			average_score = tf.Variable(0.0, 'average_score')
			n_games_played = tf.Variable(0, 'n_games_played')
			epoch_time = tf.Variable(0.0, 'epoch_time')

		played_q_value = tf.Variable(-1.0, 'played_q_value')

		tf.summary.scalar("average score per epoch", average_score, collections = ['per_epoch'])
		tf.summary.scalar("games played per epoch", n_games_played, collections = ['per_epoch'])
		tf.summary.scalar("epoch_time (s)", epoch_time, collections = ['per_epoch'])

		tf.summary.scalar("played q value", played_q_value, collections = ['per_state'])

		return average_score, n_games_played, epoch_time, played_q_value

	def tf_session_init(self):
		
		#tf.reset_default_graph()
		### initialize training graph
		self.sess = tf.Session()
		self.sess.run(self.init)
		#self.sess.graph.finalize() # make sure nothing else is added to graph

		### initialize prediction graph
		#self.sess_pred = tf.Session(graph = self.g_pred)
		#self.sess_pred.run(self.init_pred)

		### initialize tensorboard
		#self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter(self.logdir)
		self.writer.add_graph(self.sess.graph)
		#self.writer.add_graph(self.sess_pred.graph)

	#def tf_summary(self, iteration):
#
	#	self.sess.run(self.summ)
	#	self.writer.add_summary(s, iteration)

	def model_save(self, iteration):

		self.saver(self.sess, os.path.join(self.logdir, 'model.ckpt'), iteration)

	def nature_cnn(self, x_input, name):

		with tf.variable_scope("network" + name):
		# Convolutional Layers
			conv1 = tf.layers.conv2d(
			  inputs = x_input,
			  filters = 32,
			  kernel_size = [8, 8],
			  strides = (4,4),
			  padding = "valid", #valid means no padding
			  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			  bias_initializer=tf.zeros_initializer(),
			  activation = tf.nn.relu,
			  name = 'conv1' + name) #output is 20x20x32
	
			conv2 = tf.layers.conv2d(
			  inputs = conv1,
			  filters = 64,
			  kernel_size = [4, 4],
			  strides = (2,2),
			  padding = "valid", #valid means no padding
			  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			  bias_initializer=tf.zeros_initializer(),
			  activation = tf.nn.relu,
			  name = 'conv2' + name) #output is 9x9x64
	
			conv3 = tf.layers.conv2d(
			  inputs = conv2,
			  filters = 64,
			  kernel_size = [3, 3],
			  strides = (1,1),
			  padding = "valid", #valid means no padding
			  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			  bias_initializer=tf.zeros_initializer(),
			  activation = tf.nn.relu,
			  name = 'conv3' + name) #output is 7x7x64
	
			# Output layer (dense layer)
			conv3_flat = tf.reshape(conv3,[-1,7*7*64])
			FC = tf.layers.dense(
				inputs=conv3_flat, 
				units = 512, 
				kernel_initializer = tf.contrib.layers.xavier_initializer(), 
				activation = tf.nn.relu, 
				bias_initializer=tf.zeros_initializer(),
				name = 'FC' + name)
	
			output = tf.layers.dense(
				inputs = FC, 
				units = 3,  
				kernel_initializer = tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.zeros_initializer(),
				name = 'output' + name)
	
			best_action = tf.argmax(input = output, axis = 1)
			max_Q_value = tf.reduce_max(output, axis = 1, name = 'Q_max' + name)

		return output, best_action, max_Q_value

	def loss_func(self):

		with tf.name_scope("loss"):
			onehot_actions = tf.one_hot(indices = self.a, depth=3) #batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
			onehot_actions = tf.transpose(onehot_actions)
			Q_values = tf.diag_part(tf.matmul(self.output, onehot_actions))
			loss = tf.reduce_mean(tf.square(self.y - Q_values), axis = 0, name = 'loss')

			tf.summary.scalar("loss", loss, collections = ['per_state'])

		return loss

	def train(self):

		with tf.name_scope("train"):
			optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, momentum = 0.95, epsilon = 0.01)
		
			global_step = tf.Variable(0, name='global_step', trainable=False)
			train_op = optimizer.minimize(
	    	        loss = self.loss,
	    	        global_step=global_step)

		return train_op, global_step


	def mini_batch_sample(self, memory, mini_batch_size, total_transition_count):
		#Select random mini-batch of 32 transitions
		idx = np.random.randint(total_transition_count)
		mini_batch = memory[idx:idx+mini_batch_size]
	
		mini_batch_x = [x[0].astype('float32') for x in mini_batch]
		mini_batch_y = [x[3].astype('float32') for x in mini_batch]
		reward = [x[2] for x in mini_batch]
	
		mini_batch_action = [x[1] for x in mini_batch]
		return mini_batch_x, mini_batch_y, mini_batch_action, reward

	def create_y(self, reward, q_values):
		return [q_values[idx] if x is 0 else reward[idx] for idx,x in enumerate(reward)]
	
	def pick_greedy_action(self, frame_stack):
		frame_stack = [[x.astype('float32') for x in frame_stack]]
		action, q_value = self.sess.run([self.best_action, self.max_Q_value_t], feed_dict = {self.x_t : frame_stack})
		#action, q_value = self.sess.run([self.best_action_pred, self.max_Q_value_pred], feed_dict = {self.x_pred : frame_stack})
		#q_value = q_value[0]
		#self.sess.run(tf.assign(self.played_q_value, q_value))

		return action[0]
	
	def backprop(self, memory, mini_batch_size, total_transition_count):
		mini_batch_x, mini_batch_y, mini_batch_action, reward = self.mini_batch_sample(memory, mini_batch_size, total_transition_count)
		_, q_value = self.sess.run([self.best_action_pred, self.max_Q_value_pred], {self.x_pred: mini_batch_y})
	
		target_y = self.create_y(reward, q_value) # select reward as y if episode had ended
	
		feed_dict_train = {
						self.x_t : mini_batch_x,
						self.y : target_y,
						self.a : mini_batch_action
						}
	
		_, global_step, s = self.sess.run([self.train_op, self.global_step, self.summ_per_state], feed_dict = feed_dict_train)
		#_, loss_value, global_step = self.sess_t.run([self.train_op, self.loss, self.global_step], feed_dict = feed_dict_train)

		self.writer.add_summary(s, global_step)
	
		return global_step

	def update_nn(self, dataset, mini_batch_size, total_transition_count, replay_start_size):
		if total_transition_count > replay_start_size:
			global_step = self.backprop(dataset, mini_batch_size, total_transition_count)
		else: # I dont know if this is a nice solution
			global_step = 0
		return global_step

	def copy_network_weights(self):
		w_to_copy = self.sess.run(self.variables_t)
		#w_assign_to = self.sess.run(self.variables_pred)
		
		self.sess.run([tf.assign(self.variables_pred[i], w_to_copy[i]) for i in range(len(w_to_copy))])
		#for i in range(len(w_to_copy)):
		#	self.sess.run(tf.assign(self.variables_pred[i], w_to_copy[i]))
	
		return

	def accumulate_epoch_stats(self, mean_score, n_games_played, time, epoch):
		
		self.sess.run([tf.assign(self.average_score, mean_score), tf.assign(self.n_games_played, n_games_played), tf.assign(self.epoch_time, time)])
		s = self.sess.run(self.summ_per_epoch)

		self.writer.add_summary(s, epoch)