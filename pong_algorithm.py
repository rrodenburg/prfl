'''
Normalize the pixel value in the state arrays
'''

import pong_lib
import numpy as np
import pygame
import random
import sys
import tensorflow as tf

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
games_to_play = 3
max_length_dataset = 1e6
learning_rate = 0.00025

#tensorflow function
def convnetwork(mode):
    x = tf.placeholder("int32",(-1,width,heigth,frame_stacks)) #input states
    y = tf.placeholder("float32",(-1)) #y-values for loss function, as described in atari paper
    a = tf.placeholder("int32",(-1)) #actions played in batch; 0: nothing, 1: up, 2: down

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs = x,
      filters = 5,
      kernel_size = [8, 8],
      strides = (4,4),
      padding = "valid", #valid means no padding
      activation = tf.nn.relu)

    # Output layer (dense layer)
    conv1_flat = tf.reshape(conv1,[-1,20*20*5])
    output = tf.layers.dense(inputs=conv1_flat, units = 3)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
        "best_action": tf.argmax(input=output, axis=1), 
        "max_Q_value": tf.reduce_max(input=output,axis=1)
      }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss function
    onehot_actions = tf.one_hot(indices=a, depth=3)#batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
    onehot_actions = tf.transpose(onehot_actions)
    Q_values = tf.diag_part(tf.matmul(output,onehot_actions))
    loss = tf.reduce_mean(tf.square(y - Q_values),axis=0)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum = 0.95,epsilon=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



### initialize dataset
dataset = []


total_transition_count = 0

n_games_played = 0
running = True
while running: #runs once through the loop per episode
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

        next_state = np.transpose(next_state,(1,2,0)) #transpose matrix to get it in suitable shape for tensorflow
        #write an entry into the dataset
        if episode_state_count > 0:
            print(episode_state_count)
            print(state.shape)
            print(next_state.shape)
            dataset.append((state, action, reward, next_state)) 
            print('dataset entry is written!')
            total_transition_count += 1
            if total_transition_count > max_length_dataset:
                dataset.pop(0)

        episode_state_count += 1
        print(len(dataset))
        
        #Set current_state to the previous state
        state = next_state
    
    if n_games_played == games_to_play:
        pygame.quit()
        running = False
            
#np.save('repl_mem', dataset)                   

sys.exit()
                
