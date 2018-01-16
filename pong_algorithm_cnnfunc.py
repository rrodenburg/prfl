'''
Normalize the pixel value in the state arrays
'''

import pong_lib
import cnn_lib
import numpy as np
import pygame
import random
import sys
import tensorflow as tf
import time

width = 84
heigth = 84

### Hyperparameter settings
epsilon = 0.9 #probability to play a random action
frame_stacks = 3
games_to_play = 100000
max_length_dataset = 1e6
learning_rate = 0.00025
replay_start_size = 500
mini_batch_size = 16
network_copy = 1000
epoch_length = 500

def add_frame(state_list, frame_count):
	#observe state
	frame = pygame.surfarray.pixels2d(game.window).astype('float')
	frame *= 1/np.amax(frame) # Normalize the pixel intensities
	state_list[frame_count] = frame.astype('int8') #save as int8 to reduce memory usage
	return state_list

def determine_action(state, epsilon, episode_state_count, total_transition_count, replay_start_size):
	if total_transition_count > replay_start_size and epsilon > random.uniform(0, 1) and episode_state_count > 0:
		action= cnn.pick_greedy_action(state) ## Let model pick next action to play, 0 = stay, 1 = up, 2 = down
		#action = np.random.randint(3)
	else:
	    action = np.random.randint(3) ## Play random action
	return action

def rolling_avg(average, new_data_point, alpha):
	average = (1 - alpha) * average + alpha * new_data_point
	return average

def state_accumulate(action, frame_stacks, episode_running):
	frame_accumulation = True
	frame_count = 0

	next_state = np.empty([frame_stacks,84,84], dtype = 'int8') #initialize array to save frames
	
	reward = 0 # initialize reward
	while frame_accumulation == True:
	    
	    #observe reward
	    if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
	        reward += game.game_loop_action_input(action)
	        
	    if reward != 0:
	        frame_accumulation = False
	        episode_running = False #if game ends, start a new game
	        next_state = np.empty([84,84,frame_stacks], dtype = 'int8') # Return empty array of correct shape such that tensorflow can process it
	        return next_state, reward, episode_running
	    
	    #observe state
	    next_state = add_frame(next_state, frame_count)
	    
	    if frame_count == (frame_stacks - 1):
	        frame_count = 0
	        frame_accumulation = False
	    
	    frame_count += 1

	next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
	return next_state, reward, episode_running

def repl_memory_insert(dataset, state, action, reward, next_state, total_transition_count, max_length_dataset):
	#idx = np.random.randint(0, total_transition_count+1)

	#dataset.insert(idx, (state, action, reward, next_state)) # insert at random position for faster sampling
	dataset.append((state, action, reward, next_state))

	total_transition_count += 1
	if total_transition_count > max_length_dataset:
	    dataset.pop(0)
	return dataset, total_transition_count

def display_stats(backprop_cycles, n_games_played, games_won, loss_value, running_score_mean, q_value_mean):
	print('backprop_cycles : {}, episodes : {} , games won : {} , loss : {} , running score avg : {} , running q_value mean : {}'.format(backprop_cycles, n_games_played, games_won, 
		loss_value, running_score_mean, q_value_mean))
	loss_value = 0
	return loss_value

def end_game_check(reward, n_games_played, games_won):
	if reward != 0:
		n_games_played += 1
	if reward == 1:
		games_won += 1
	return n_games_played, games_won

def epoch_end(epoch, epoch_time, mean_score, games_won, n_games_played):
	epoch_time = time.time() - epoch_time

	if n_games_played > 0:
		mean_score = games_won / n_games_played

	cnn.accumulate_epoch_stats(mean_score, n_games_played, epoch_time, epoch)
	epoch += 1

	print('epoch {}: epoch time {}, n_games_played {} , games_won {}, average_score {}'.format(epoch, epoch_time, n_games_played, games_won, mean_score))

	# Reset stats
	epoch_time = time.time()
	games_won = 0
	n_games_played = 0

	return epoch, epoch_time, mean_score, games_won, n_games_played

def rf_initialization():
	### initialize dataset
	dataset = []
	total_transition_count = 0
	
	# stats initialization
	n_games_played = 0
	games_won = 0
	mean_score = 0
	
	epoch = 0
	epoch_time = time.time()
	
	backprop_cycles = 0 

	return dataset, total_transition_count, n_games_played, games_won, mean_score, epoch, epoch_time, backprop_cycles

#### Initialize Pong
game = pong_lib.Pong( 
                number_of_players = 1, 
                width = width, 
                heigth = heigth, 
                ball_radius = 2, 
                pad_width = 4, 
                pad_heigth = 14, 
                pad_velocity = 8, 
                pad_velocity_ai = 2,
                DistPadWall = 4,
                ball_velocity = 0.7)
game.game_init()

#### Initialize tensorflow graph
cnn = cnn_lib.Network(
					width = width,
					heigth = heigth,
					frame_stacks = frame_stacks,
					learning_rate = learning_rate,
					mini_batch_size = mini_batch_size,
					network_name = 'nature_cnn',
					trainer_name = 'adam'
					)

######### START RF LEARNING LOOP ######

# Initialize reninforcement learning dataset, and stats.

dataset, total_transition_count, n_games_played, games_won, mean_score, epoch, epoch_time, backprop_cycles = rf_initialization()

#initialize tensorflow settings and variables
cnn.tf_session_init()

running = True
while running: 
    
    episode_running = True

    game.initialize_ball_pad_positions()
    
    episode_state_count = 0 #counts the state number of the current episode
    
    while episode_running == True: #runs once through the loop per episode

        if episode_state_count == 0: # play no action during first frames to initialize
        	action = 0 

        	next_state, reward, episode_running = state_accumulate(action, frame_stacks, episode_running)
        
        #write an entry into the dataset, do backprop and print stats
        else:
			## Pick action following greedy policy; 1 action per episode
        	action = determine_action(state, epsilon, episode_state_count, total_transition_count, replay_start_size)

        	next_state, reward, episode_running = state_accumulate(action, frame_stacks, episode_running)
	
    		# Accumulate statistics
        	n_games_played, games_won = end_game_check(reward, n_games_played, games_won)

        	backprop_cycles = cnn.update_nn(dataset, mini_batch_size, total_transition_count, replay_start_size)

        	dataset, total_transition_count = repl_memory_insert(dataset, state, action, reward, next_state, total_transition_count, max_length_dataset)


        	if (backprop_cycles % network_copy == 0) and (backprop_cycles > 0):
        		cnn.copy_network_weights()

        	if (backprop_cycles % epoch_length == 0) and (backprop_cycles > 0):
        		#Calculate states, send stats to tensorboard and print stats in terminal
        		epoch, epoch_time, mean_score, games_won, n_games_played = epoch_end(epoch, epoch_time, mean_score, games_won, n_games_played)

        	if (backprop_cycles % (epoch_length*20) == 0) and (backprop_cycles > 0):
        		# Save learned variables to disk
        		cnn.model_save(epoch)

        episode_state_count += 1
        
        #Set current_state to the previous state
        state = next_state
    
    if n_games_played == games_to_play:
        pygame.quit()
        running = False                 

sys.exit()
                
