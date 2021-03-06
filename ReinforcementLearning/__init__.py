import Pong
import models
import time
import numpy as np
import pygame
import random
import os
from pathlib import Path

def make_logdir(logdir):
	subfolder = 1
	logfolder = Path(logdir) / str(subfolder)
	while os.path.isdir(logfolder) == True:
		subfolder += 1
		logfolder = Path(logdir) / str(subfolder)
	
	if not os.path.exists(logfolder):
	    os.makedirs(logfolder)
	
	print('Data is saved in :', logfolder)

	return logfolder

def write_settings_logfile(logfolder, x):
	
	# write file in log directory
	with open(logfolder / 'settings.txt', 'w+') as f:
		for key, val in x.items():
			f.write(key + ':' + str(val) + '\n')

	return

class RF(object):

	def __init__(
				self,
				epsilon = 0.9,
				epsilon_decay = 40000,
				frame_stacks = 4,
				model = None,
				game = None,
				gui = False
				):

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.frame_stacks = frame_stacks
		self.model = model
		self.game = game
		self.gui = gui
		

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
		
		if self.epsilon_decay == 0:
			current_epsilon = self.epsilon
		else:
			current_epsilon = (self.epsilon + (1 - self.epsilon) * (min(1,backprop_cycles / self.epsilon_decay)))

		if total_transition_count > replay_start_size and  current_epsilon > random.uniform(0, 1) and episode_state_count > 0:
			action= self.model.pick_greedy_action(state) ## Let model pick next action to play, 0 = stay, 1 = up, 2 = down
		else:
		    action = np.random.randint(3) ## Play random action
		return action
	
	def state_accumulate(self, action, episode_running):
		frame_accumulation = True
		frame_count = 0
	
		next_state = np.empty([self.frame_stacks,84,84], dtype = 'int8') #initialize array to save frames
		
		reward = 0 # initialize reward
		while frame_accumulation == True:
		    
		    #observe reward
		    if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
		    	if self.gui == True:
		        	reward += self.game.game_loop_action_input(action)
		    	else:
		    		reward_temp, next_state[frame_count] = self.game.game_loop_action_input_np(action)
		    		reward += reward_temp

		        
		    if reward != 0:
		        frame_accumulation = False
		        episode_running = False #if game ends, start a new game
		        next_state = np.empty([84,84,self.frame_stacks], dtype = 'int8') # Return empty array of correct shape such that tensorflow can process it
		        return next_state, reward, episode_running
		    
		    #observe state
		    if self.gui == True:
		    	next_state = self.add_frame(next_state, frame_count)
		    
		    if frame_count == (self.frame_stacks - 1):
		        frame_count = 0
		        frame_accumulation = False
		    
		    frame_count += 1
	
		next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
		return next_state, reward, episode_running

	#def state_accumulate_np(self, action, episode_running):
	#	frame_accumulation = True
	#	frame_count = 0
	#
	#	next_state = np.empty([self.frame_stacks,84,84], dtype = 'int8') #initialize array to save frames
	#	
	#	reward = 0 # initialize reward
	#	while frame_accumulation == True:
	#	    
	#	    #observe reward and state
	#	    if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
	#	        reward_temp, next_state[frame_count] = self.game.game_loop_action_input_np(action)
	#	        reward += reward_temp
	#	        
	#	    if reward != 0:
	#	        frame_accumulation = False
	#	        episode_running = False #if game ends, start a new game
	#	        next_state = np.empty([84,84,self.frame_stacks], dtype = 'int8') # Return empty array of correct shape such that tensorflow can process it
	#	        return next_state, reward, episode_running
	#	    
	#	    #observe state
	#	    #next_state = self.add_frame(next_state, frame_count)
	#	    
	#	    if frame_count == (self.frame_stacks - 1):
	#	        frame_count = 0
	#	        frame_accumulation = False
	#	    
	#	    frame_count += 1
	#
	#	next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
	#	return next_state, reward, episode_running
	
	def repl_memory_insert(self, dataset, state, action, reward, next_state, total_transition_count, max_length_dataset):
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
	
		self.model.accumulate_epoch_stats(mean_score, n_games_played, epoch_time, epoch)
		epoch += 1
	
		print('epoch {}: epoch time {}, n_games_played {} , games_won {}, average_score {}'.format(epoch, epoch_time, n_games_played, games_won, mean_score))
	
		# Reset stats
		epoch_time = time.time()
		games_won = 0
		n_games_played = 0
	
		return epoch, epoch_time, mean_score, games_won, n_games_played
	
	