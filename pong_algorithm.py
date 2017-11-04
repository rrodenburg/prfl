'''
Normalize the pixel value in the state arrays
'''

import pong_lib
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
import sys

game = pong_lib.Pong( 
                number_of_players = 1, 
                width = 84, 
                heigth = 84, 
                ball_radius = 2, 
                pad_width = 4, 
                pad_heigth = 14, 
                pad_velocity = 4, 
                pad_velocity_ai = 1,
                DistPadWall = 4,
                ball_velocity = 0.3)
game.game_init()



### Hyperparameter settings
epsilon = 0.9 #probability to play a random action
frame_stacks = 4
games_to_play = 1


### initialize dataset
dataset = []


total_frame_count = 0

n_games_played = 0
running = True
while running:
    #play as long as game lasts
    episode_running = True
    print('a new pong game has started!')
    game.initialize_ball_pad_positions()
    
    #initialize a current state, the transition from the initialed state will later be deleted from the dataset
    
    state = np.zeros([frame_stacks,84,84], dtype = 'int')
    
    episode_state_count = 0 #counts the state number of the current episode
    
    while episode_running == True:
        
        ## Pick action following greedy policy; 1 action per episode
        if epsilon > random.uniform(0, 1):
            action = random.choice(['stay','up','down']) ## Let model pick next action to play
        else:
            action = random.choice(['stay','up','down']) ## Play random action
            
        #initialize array to save frames
        next_state = np.empty([frame_stacks,84,84], dtype = 'int')
        
        # Run action and observe reward and next state
        frame_accumulation = True
        frame_count = 0
        
        reward = 0 # initialize reward
        while frame_accumulation == True:
            
            #observe reward
            if reward == 0: # does not allow to obtain a reward of +2 or -2 if game termination does not occur instantly
                reward += game.game_loop_action_input(action)
                
            if reward != 0:
                print('someone won! game must terminate!')
                n_games_played += 1
                frame_accumulation = False
                episode_running = False #if game ends, start a new game
            
            #observe state
            frame = pygame.surfarray.pixels2d(game.window).astype('float')
            frame *= 255/np.amax(frame) # Normalize the pixel intensities
            next_state[frame_count] = frame.astype('int')
            
            if frame_count == 3:
                frame_count = 0
                frame_accumulation = False
            
            frame_count += 1
        
        #write an entry into the dataset
        if episode_state_count > 0:
            dataset.append((state, action, reward, next_state))
            
        episode_state_count += 1
        print('dataset entry is written!')
        
        #Set current_state to the previous state
        state = next_state
    
    if n_games_played == games_to_play:
        pygame.quit()
        running = False
            
np.save('repl_mem', dataset)                   
np.set_printoptions(threshold=np.nan)            
print(dataset[5][0])
print(np.amax(dataset[5][3][2]))
plt.imshow(dataset[5][3][0], cmap = 'Greys')
#plt.imshow(dataset[5][0][1], cmap = 'Greys')
#plt.show()    

sys.exit()
                
                    
                
        
        
   # (reward, running) = game.game_loop_action_input(previous_action, running)
   # previous_reward += reward
   # if reward != 0:
   #     print(reward)
   # current_state = np.concatenate((current_state,[pygame.surfarray.pixels2d(game.window)]))
   # frame_count += 1
   # total_frame_count += 1
   # if frame_count == 3:
   #     dataset.append((previous_state, previous_action, previous_reward, current_state))
   #     previous_state = current_state
   #     current_state = np.empty([1,84,84], dtype = 'int')
   #     previous_reward = 0
   #     frame_count = 0
   #     
   #     
        
    #if total_frame_count > 100:
    #    running = False
        

#print(dataset[5][0].shape)
#print(np.amax(dataset[5][0][2]))
#plt.imshow(dataset[5][0][3], cmap = 'Greys')
#plt.imshow(dataset[5][0][1], cmap = 'Greys')
#plt.show()

        
