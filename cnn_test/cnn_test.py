import numpy as np
import tensorflow as tf
import random

memory = np.load('../../data/repl_mem.npy')
#memory = memory[:32]

print(len(memory))
print(np.take(memory, indices = 0, axis = 1)[0].shape)

### Define neural network ### https://www.tensorflow.org/tutorials/layers

learning_rate = 0.0001
frame_stacks = 4
width = 84
heigth = 84
mini_batch_size = 32

tf.reset_default_graph()

g_1 = tf.Graph()
with g_1.as_default():

	x = tf.placeholder("float32",(None,width,heigth,frame_stacks)) #input states
	y = tf.placeholder("float32",(None)) #y-values for loss function, as described in atari paper
	a = tf.placeholder("int32",(None)) #actions played in batch; 0: nothing, 1: up, 2: down

	# Convolutional Layers
	#conv1 = tf.layers.conv2d(
	#  inputs = x,
	#  filters = 32,
	#  kernel_size = [8, 8],
	#  strides = (4,4),
	#  padding = "valid", #valid means no padding
	#  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
	#  bias_initializer=tf.zeros_initializer(),
	#  activation = tf.nn.relu) #output is 20x20x32
	weigths_1 = {
	'conv1_w' : tf.get_variable("weights", [8,8,4,32],
	    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
	'conv1_b' : tf.get_variable("biases", [32],
	    initializer=tf.constant_initializer(0.0))
	}

	#weights_1 = tf.get_variable("weights", [8,8,4,32],
	#    initializer=tf.contrib.layers.xavier_initializer_conv2d())
	## Create variable named "biases".
	#biases_1 = tf.get_variable("biases", [32],
	#    initializer=tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(x, weigths_1['conv1_w'],
	    strides=[1, 4, 4, 1], padding='SAME')
	conv1 = tf.nn.relu(conv + weigths_1['conv1_b'])

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

	#loss function
	onehot_actions = tf.one_hot(indices=a, depth=3) #batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
	onehot_actions = tf.transpose(onehot_actions)
	Q_values = tf.diag_part(tf.matmul(output,onehot_actions))
	loss = tf.reduce_mean(tf.square(y - Q_values), axis = 0)

	optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum = 0.95, epsilon = 0.01)

	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(
	            loss=loss,
	            global_step=global_step)

	init_1 = tf.global_variables_initializer()

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.zeros_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 4, 4, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

#### GRAPH 2 prediction only ####

g_2 = tf.Graph()
with g_2.as_default():

	x_2 = tf.placeholder("float32",(None,width,heigth,frame_stacks)) #input states

	# Convolutional Layers
	#conv1 = conv_relu(x_2, [8,8,4,32], [32])

	weigths = {
	'conv1_w' : tf.get_variable("weights", [8,8,4,32],
	    initializer=tf.zeros_initializer()),
	'conv1_b' : tf.get_variable("biases", [32],
	    initializer=tf.constant_initializer(0.0))
	}

	#weights = tf.get_variable("weights", [8,8,4,32],
	#    initializer=tf.zeros_initializer())
	## Create variable named "biases".
	#biases = tf.get_variable("biases", [32],
	#    initializer=tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(x_2, weigths['conv1_w'],
	    strides=[1, 4, 4, 1], padding='SAME')
	conv1 = tf.nn.relu(conv + weigths['conv1_b'])

	#conv1 = tf.layers.conv2d(
	#  inputs = x_2,
	#  filters = 32,
	#  kernel_size = [8, 8],
	#  strides = (4,4),
	#  padding = "valid", #valid means no padding
	#  kernel_initializer = tf.zeros_initializer(),
	#  bias_initializer=tf.zeros_initializer(),
	#  activation = tf.nn.relu) #output is 20x20x32

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
	best_action_2 = tf.argmax(input = output, axis = 1)
	max_Q_value_2 = tf.reduce_max(output, axis = 1, name = 'Q_max')

	init_2 = tf.global_variables_initializer()

def mini_batch_sample(memory, mini_bach_size):
#Select random mini-batch of 32 transitions
	idx = np.random.randint(len(memory), size = mini_bach_size)
	mini_batch = memory[idx]

	mini_batch_x = [x[0].astype('float32') for x in mini_batch]
	mini_batch_y = [x[3].astype('float32') for x in mini_batch]
	reward = [x[2] for x in mini_batch]

	mini_batch_action = [x[1] for x in mini_batch]
	return mini_batch_x, mini_batch_y, mini_batch_action, reward

def create_y(reward, q_values):
	return [q_values[idx] if x is 0 else reward[idx] for idx,x in enumerate(reward)]

def determine_action(frame_stack):
	frame_stack = [[x.astype('float32') for x in predict_x]]
	action = sess_1.run(best_action, feed_dict = {x : frame_stack})
	return action[0]

def backprop(memory, mini_batch_size):
	mini_batch_x, mini_batch_y, mini_batch_action, reward = mini_batch_sample(memory, mini_batch_size)
	_, q_value = sess_2.run([best_action_2, max_Q_value_2], {x_2: mini_batch_y})
	#print(q_value[0])

	target_y = create_y(reward, q_value) # select reward as y if episode had ended

	feed_dict_train = {
					x : mini_batch_x,
					y : target_y,
					a : mini_batch_action
					}

	_, loss_value = sess_1.run([train_op, loss], feed_dict = feed_dict_train)

	return loss_value

def copy_weights():
	#tf.contrib.copy_graph.copy_variable_to_graph()
	with g_1.as_default():
		train_1_var = tf.trainable_variables()
	w_to_copy = sess_1.run(train_1_var)

	#print(w_to_copy[0])

	with g_2.as_default():
		train_var = tf.trainable_variables()

	for i in range(len(train_var)):
		#ass = tf.assign(train_var[i], w_to_copy[i])
		sess_2.run(tf.assign(train_var[i], w_to_copy[i]))
		#print('{} is copied'.format(i))
	return


sess_1 = tf.Session(graph = g_1)
sess_1.run(init_1)

sess_2 = tf.Session(graph = g_2)
sess_2.run(init_2)


mini_batch_x, mini_batch_y, mini_batch_action, reward = mini_batch_sample(memory, mini_batch_size)
_, q_value = sess_1.run([best_action, max_Q_value], {x: mini_batch_y})

target_y = create_y(reward, q_value)

loss_value = 0

for i in range(10000):
	
	#action = determine_action(predict_x)

	loss_value += backprop(memory, mini_batch_size)
	
	if i % 100 == 0 and i > 0: 
		print('loss after ', i, ' backprop cycles = ', loss_value)
		loss_value = 0
		copy_weights()


sess.close()

#copy_ops = [variables[ix+len(variables)//2].assign(var.value()) for ix, var in enumerate(variables[0:len(variables)//2])]
#
#sess2 = tf.Session()
#sess2.run(init)
#map(lambda x: sess.run(x), copy_ops)