'''
Normalize the pixel value in the state arrays
'''

import pong_lib
import numpy as np
import pygame
import random
import sys
import tensorflow as tf

width = 84
heigth = 84

game = pong_lib.Pong( 
                number_of_players = 1, 
                width = width, 
                heigth = heigth, 
                ball_radius = 2, 
                pad_width = 4, 
                pad_heigth = 14, 
                pad_velocity = 8, 
                pad_velocity_ai = 4,
                DistPadWall = 4,
                ball_velocity = 1.5)
game.game_init()

### Hyperparameter settings
epsilon = 0.9 #probability to play a random action
frame_stacks = 2
games_to_play = 10000
max_length_dataset = 1e6
learning_rate = 0.00025
replay_start_size = 1000
mini_batch_size = 16

#tensorflow function
tf.reset_default_graph()

x = tf.placeholder("float32",(None,width,heigth,frame_stacks)) #input states
y = tf.placeholder("float32",(None)) #y-values for loss function, as described in atari paper
a = tf.placeholder("int32",(None)) #actions played in batch; 0: nothing, 1: up, 2: down

# Convolutional Layers
conv1 = tf.layers.conv2d(
  inputs = x,
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

def mini_batch_sample(memory, mini_batch_size):
#Select random mini-batch of 32 transitions
	idx = np.random.randint(len(memory), size = mini_batch_size)
	mini_batch = [memory[i] for i in idx]

	mini_batch_x = [x[0].astype('float32') for x in mini_batch]
	mini_batch_y = [x[3].astype('float32') for x in mini_batch]
	reward = [x[2] for x in mini_batch]

	mini_batch_action = [x[1] for x in mini_batch]
	return mini_batch_x, mini_batch_y, mini_batch_action, reward

def create_y(reward, q_values):
	return [q_values[idx] if x is 0 else reward[idx] for idx,x in enumerate(reward)]

def determine_action(frame_stack):
	#frame_stack = np.transpose(frame_stack, (1, 0, 2))
	frame_stack = [[x.astype('float32') for x in frame_stack]]
	action, q_value = sess.run([best_action, max_Q_value], feed_dict = {x : frame_stack})
	return action[0], q_value[0]

def backprop(memory, mini_batch_size):
	mini_batch_x, mini_batch_y, mini_batch_action, reward = mini_batch_sample(memory, mini_batch_size)
	_, q_value = sess.run([best_action, max_Q_value], {x: mini_batch_y})

	target_y = create_y(reward, q_value) # select reward as y if episode had ended

	feed_dict_train = {
					x : mini_batch_x,
					y : target_y,
					a : mini_batch_action
					}

	_, loss_value = sess.run([train_op, loss], feed_dict = feed_dict_train)

	return loss_value



### initialize dataset
dataset = []


total_transition_count = 0

n_games_played = 0
games_won = 0 

# stats initialization
running_score_mean = 0.0
running_score_alpha = 0.01
running_score_mean_list = []

q_value_mean = 0.0
q_value_alpha = 0.01

loss_value = 0.0
backprop_cycles = 0 

#initialize tensorflow settings and variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

running = True
while running: #runs once through the loop per episode
    #play as long as game lasts
    episode_running = True
    #print('a new pong game has started!')
    game.initialize_ball_pad_positions()
    
    #initialize a current state, the transition from the initialed state will later be deleted from the dataset
    
    state = np.zeros([frame_stacks,84,84], dtype = 'int8')
    
    episode_state_count = 0 #counts the state number of the current episode
    
    while episode_running == True:
        
        ## Pick action following greedy policy; 1 action per episode
        if total_transition_count > replay_start_size and epsilon > random.uniform(0, 1) and episode_state_count > 0:
	        	action, q_value = determine_action(state)## Let model pick next action to play, 0 = stay, 1 = up, 2 = down
	        	#print('network action :' , action)
	        	q_value_mean = (1 - q_value_alpha) * q_value_mean + q_value_alpha * q_value
        else:
            action = np.random.randint(3) ## Play random action
    
        #initialize array to save frames
        next_state = np.empty([frame_stacks,84,84], dtype = 'int8')
        
        # Run action and observe reward and next state
        frame_accumulation = True
        frame_count = 0
        
        reward = 0 # initialize reward
        while frame_accumulation == True:
            
            #observe reward
            if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
                reward += game.game_loop_action_input(action)
                
            if reward != 0:
                #print('someone won! game must terminate!')
                n_games_played += 1
                if reward == 1:
                	games_won += 1
                running_score_mean = (1 - running_score_alpha) * running_score_mean + running_score_alpha * reward #Checck if correct!
                running_score_mean_list.append(running_score_mean)
                frame_accumulation = False
                episode_running = False #if game ends, start a new game
            
            #observe state
            frame = pygame.surfarray.pixels2d(game.window).astype('float')
            frame *= 1/np.amax(frame) # Normalize the pixel intensities
            next_state[frame_count] = frame.astype('int8') #save as int8 to reduce memory usage
            
            if frame_count == (frame_stacks - 1):
                frame_count = 0
                frame_accumulation = False
            
            frame_count += 1

        next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
        
        #write an entry into the dataset, do backprop and print stats
        if episode_state_count > 0:
        	dataset.append((state, action, reward, next_state)) 
        	total_transition_count += 1
        	if total_transition_count > max_length_dataset:
        	    dataset.pop(0)

        	# perform backprop
        	if total_transition_count > replay_start_size:
        		loss_value += backprop(dataset, mini_batch_size)
        		backprop_cycles += 1
        		#print('backprop	done')

        	# display statistics
        	if total_transition_count % 200 == 0:
        		print('backprop_cycles : {}, episodes : {} , games won : {} , loss : {} , running score avg : {} , running q_value mean : {}'.format(backprop_cycles, n_games_played, games_won, 
        			loss_value, running_score_mean, q_value_mean))
        		loss_value = 0

        episode_state_count += 1
        #print(len(dataset))
        
        #Set current_state to the previous state
        state = next_state
    
    if n_games_played == games_to_play:
        pygame.quit()
        running = False
            
#np.save('../data/repl_mem', dataset)                   

sys.exit()
                
