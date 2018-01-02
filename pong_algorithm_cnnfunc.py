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
frame_stacks = 4
games_to_play = 10000
max_length_dataset = 1e6
learning_rate = 0.00025
replay_start_size = 50
mini_batch_size = 16

cnn = cnn_lib.Network(
					width = width,
					heigth = heigth,
					frame_stacks = frame_stacks,
					learning_rate = learning_rate,
					mini_batch_size = mini_batch_size,
					network_name = 'nature_cnn'
					)

def add_frame(state_list, frame_count):
	#observe state
	frame = pygame.surfarray.pixels2d(game.window).astype('float')
	frame *= 1/np.amax(frame) # Normalize the pixel intensities
	state_list[frame_count] = frame.astype('int8') #save as int8 to reduce memory usage
	return state_list

def determine_action(state, epsilon, episode_state_count, total_transition_count, replay_start_size):
	if total_transition_count > replay_start_size and epsilon > random.uniform(0, 1) and episode_state_count > 0:
		action, q_value = cnn.pick_greedy_action(state) ## Let model pick next action to play, 0 = stay, 1 = up, 2 = down
	else:
	    action = np.random.randint(3) ## Play random action
	    q_value = None
	return action, q_value

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

def repl_memory_append(dataset, state, action, reward, next_state, total_transition_count, max_length_dataset):
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

def accumulate_stats(reward, n_games_played, games_won, running_score_mean, running_score_alpha, q_value_mean, q_value, q_value_alpha):
	if reward != 0:
		n_games_played += 1
		running_score_mean = rolling_avg(running_score_mean, reward, running_score_alpha)
	if reward == 1:
		games_won += 1
	if q_value is not None:
		q_value_mean = rolling_avg(q_value_mean, q_value, q_value_alpha)
	return n_games_played, games_won, running_score_mean, q_value_mean


### initialize dataset
dataset = []
total_transition_count = 0

n_games_played = 0
games_won = 0 

# stats initialization
running_score_mean = 0.0
running_score_alpha = 0.01

q_value_mean = 0.0
q_value_alpha = 0.01

loss_value = 0.0
backprop_cycles = 0 

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
        	action, q_value = determine_action(state, epsilon, episode_state_count, total_transition_count, replay_start_size)
        	
        	next_state, reward, episode_running = state_accumulate(action, frame_stacks, episode_running)
	
    		# Accumulate statistics
        	n_games_played, games_won, running_score_mean, q_value_mean = accumulate_stats(reward, n_games_played, games_won, running_score_mean, running_score_alpha, q_value_mean, q_value, q_value_alpha)

        	loss_value, backprop_cycles = cnn.update_nn(dataset, mini_batch_size, loss_value, backprop_cycles, total_transition_count, replay_start_size)

        	dataset, total_transition_count = repl_memory_append(dataset, state, action, reward, next_state, total_transition_count, max_length_dataset)

        	# display statistics
        	if total_transition_count % 200 == 0:
        		loss_value = display_stats(backprop_cycles, n_games_played, games_won, loss_value, running_score_mean, q_value_mean)

        episode_state_count += 1
        
        #Set current_state to the previous state
        state = next_state
    
    if n_games_played == games_to_play:
        pygame.quit()
        running = False                 

sys.exit()
                
