
import pong_lib
import cnn_lib
import time
import numpy as np
import pygame
import random
import os

def make_logdir(logdir):
	subfolder = 1
	logfolder = logdir + str(subfolder)
	while os.path.isdir(logfolder) == True:
		subfolder += 1
		logfolder = logdir + str(subfolder)
	
	if not os.path.exists(logfolder):
	    os.makedirs(logfolder)
	
	print('Data is saved in :', logfolder)

	return logfolder

def write_settings_logfile(logfolder, network_name, trainer_name, frame_stacks, max_epochs, max_length_dataset, learning_rate, replay_start_size, mini_batch_size, network_copy, epoch_length, epsilon, epsilon_decay):

	parameter_list = [
					'network_name : {}'.format(network_name),
					'trainer_name : {}'.format(trainer_name),
					'frame_stacks : {}'.format(frame_stacks),
					'max epochs : {}'.format(max_epochs),
					'max_length_dataset : {}'.format(max_length_dataset),
					'learning_rate : {}'.format(learning_rate),
					'replay_start_size : {}'.format(replay_start_size),
					'mini_batch_size : {}'.format(mini_batch_size),
					'network_copy : {}'.format(network_copy),
					'epoch_length : {}'.format(epoch_length),
					'epsilon : {}'.format(epsilon),
					'decay epsilon over x backprop cycles : {}'.format(epsilon_decay)
					]
	
	# write file in log directory
	with open(logfolder + '/settings.txt', 'w+') as f:
		[f.write(i + '\n') for i in parameter_list]

	return

class RF(object):

	def __init__(
				self,
				epsilon = 0.9,
				epsilon_decay = 40000,
				cnn = None,
				game = None
				):

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.cnn = cnn
		self.game = game
		

	def rf_initialization(self):
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
	
	def add_frame(self, state_list, frame_count):
		#observe state
		frame = pygame.surfarray.pixels2d(self.game.window).astype('float')
		frame *= 1/np.amax(frame) # Normalize the pixel intensities
		state_list[frame_count] = frame.astype('int8') #save as int8 to reduce memory usage
		return state_list
	
	def determine_action(self, state, episode_state_count, total_transition_count, replay_start_size, backprop_cycles):
		current_epsilon = (self.epsilon + (1 - self.epsilon) * (min(1,backprop_cycles / self.epsilon_decay)))
		if total_transition_count > replay_start_size and  current_epsilon > random.uniform(0, 1) and episode_state_count > 0:
			action= self.cnn.pick_greedy_action(state) ## Let model pick next action to play, 0 = stay, 1 = up, 2 = down
		else:
		    action = np.random.randint(3) ## Play random action
		return action
	
	def state_accumulate(self, action, frame_stacks, episode_running):
		frame_accumulation = True
		frame_count = 0
	
		next_state = np.empty([frame_stacks,84,84], dtype = 'int8') #initialize array to save frames
		
		reward = 0 # initialize reward
		while frame_accumulation == True:
		    
		    #observe reward
		    if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
		        reward += self.game.game_loop_action_input(action)
		        
		    if reward != 0:
		        frame_accumulation = False
		        episode_running = False #if game ends, start a new game
		        next_state = np.empty([84,84,frame_stacks], dtype = 'int8') # Return empty array of correct shape such that tensorflow can process it
		        return next_state, reward, episode_running
		    
		    #observe state
		    next_state = self.add_frame(next_state, frame_count)
		    
		    if frame_count == (frame_stacks - 1):
		        frame_count = 0
		        frame_accumulation = False
		    
		    frame_count += 1
	
		next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
		return next_state, reward, episode_running
	
	def repl_memory_insert(self, dataset, state, action, reward, next_state, total_transition_count, max_length_dataset):
		#idx = np.random.randint(0, total_transition_count+1)
	
		#dataset.insert(idx, (state, action, reward, next_state)) # insert at random position for faster sampling
		dataset.append((state, action, reward, next_state))
	
		total_transition_count += 1
		if total_transition_count > max_length_dataset:
		    dataset.pop(0)
		return dataset, total_transition_count
	
	def end_game_check(self, reward, n_games_played, games_won):
		if reward != 0:
			n_games_played += 1
		if reward == 1:
			games_won += 1
		return n_games_played, games_won
	
	def epoch_end(self, epoch, epoch_time, mean_score, games_won, n_games_played):
		epoch_time = time.time() - epoch_time
	
		if n_games_played > 0:
			mean_score = games_won / n_games_played
	
		self.cnn.accumulate_epoch_stats(mean_score, n_games_played, epoch_time, epoch)
		epoch += 1
	
		print('epoch {}: epoch time {}, n_games_played {} , games_won {}, average_score {}'.format(epoch, epoch_time, n_games_played, games_won, mean_score))
	
		# Reset stats
		epoch_time = time.time()
		games_won = 0
		n_games_played = 0
	
		return epoch, epoch_time, mean_score, games_won, n_games_played
	
	