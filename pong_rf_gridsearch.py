import pong_lib
import cnn_lib
import rf_lib
#from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import ParameterGrid

logdir = '../lol_test/'

# Network settings
param_grid = {
                'network_name' : ['nature_cnn'],
                'trainer_name' : ['adam'],
                'epsilon' : [0.9], #probability to play a random action
                'epsilon_decay' : [500 * 250], # number of backprop cycles to linear scale epsilon to 1. Put 0 for no decay
                'frame_stacks' : [3,1],
                'max_epochs' : [750],
                'max_length_dataset' : [10e6], # Maximum size of the replay network dataset
                'learning_rate' : [0.00025],
                'replay_start_size' : [50], # Number of states of random play, before network starts action picking
                'mini_batch_size' : [16],
                'network_copy' : [1000], # Number of backprop cycles needed to copy network weigths from training to target network
                'epoch_length' : [500],
                'gui' : [True]
                }

grid = ParameterGrid(param_grid)

def run_training(x, logdir = logdir):

    max_epochs, max_length_dataset, replay_start_size, network_copy, epoch_length = x['max_epochs'], x['max_length_dataset'], x['replay_start_size'], x['network_copy'], x['epoch_length']

    #### Create logfolder and write settings file
    logfolder = rf_lib.make_logdir(logdir)
    rf_lib.write_settings_logfile(logfolder, x)
    
    # Window size of the game
    width = 84
    heigth = 84

    #### Initialize Pong
    game = pong_lib.Pong( 
                    number_of_players = 1, 
                    width = width, 
                    heigth = heigth, 
                    ball_radius = 2, 
                    pad_width = 4, 
                    pad_heigth = 14, 
                    pad_velocity = 1, 
                    pad_velocity_ai = 3,
                    DistPadWall = 4,
                    ball_velocity = 0.4,
                    speed_increase = 0.1,
                    gui = x['gui']
                    )
    
    #### Initialize tensorflow graph
    cnn = cnn_lib.Network(
    					width = width,
    					heigth = heigth,
    					frame_stacks = x['frame_stacks'],
    					learning_rate = x['learning_rate'],
    					mini_batch_size = x['mini_batch_size'],
    					network_name = x['network_name'],
    					trainer_name = x['trainer_name'],
    					logdir = logfolder
    					)
    
    #### Initiazlize rf library using initialized network and game
    rf = rf_lib.RF(
    				epsilon = x['epsilon'],
                    frame_stacks = x['frame_stacks'],
                    epsilon_decay = x['epsilon_decay'],
                    cnn = cnn,
                    game = game,
                    gui = x['gui']
    				)
    
    
    
    ######### START RF LEARNING LOOP ######
    
    # Initialize reninforcement learning dataset, and stats.
    
    dataset, total_transition_count, n_games_played, games_won, mean_score, epoch, epoch_time, backprop_cycles = rf.rf_initialization()
    
    #initialize tensorflow settings and variables
    cnn.tf_session_init()
    
    # Initialize pong game
    game.game_init()
    
    running = True
    while running: 
        
        episode_running = True
    
        game.initialize_ball_pad_positions()
        
        episode_state_count = 0 #counts the state number of the current episode
        
        while episode_running == True: #runs once through the loop per episode
    
            if episode_state_count == 0: # play no action during first frames to initialize
            	action = 0 
    
            	next_state, reward, episode_running = rf.state_accumulate(action, episode_running)
            
            #write an entry into the dataset, do backprop and print stats
            else:
    			## Pick action following greedy policy; 1 action per episode
            	action = rf.determine_action(state, episode_state_count, total_transition_count, replay_start_size, backprop_cycles)
    
            	next_state, reward, episode_running = rf.state_accumulate(action, episode_running)
    	
        		# Accumulate statistics
            	n_games_played, games_won = rf.end_game_check(reward, n_games_played, games_won)
    
            	backprop_cycles = cnn.update_nn(dataset, total_transition_count, replay_start_size)
    
            	dataset, total_transition_count = rf.repl_memory_insert(dataset, state, action, reward, next_state, total_transition_count, max_length_dataset)
    
    
            	if (backprop_cycles % network_copy == 0) and (backprop_cycles > 0):
            		cnn.copy_network_weights()
    
            	if (backprop_cycles % epoch_length == 0) and (backprop_cycles > 0):
            		#Calculate states, send stats to tensorboard and print stats in terminal
            		epoch, epoch_time, mean_score, games_won, n_games_played = rf.epoch_end(epoch, epoch_time, mean_score, games_won, n_games_played)
    
            	if (backprop_cycles % (epoch_length * 20) == 0) and (backprop_cycles > 0):
            		# Save learned variables to disk
            		cnn.model_save(backprop_cycles)
    
            	### Terminate training after max epochs
            	if (backprop_cycles / epoch_length) >= max_epochs:
                    cnn.model_save(backprop_cycles)
                    running = False
                    episode_running = False
    
            episode_state_count += 1
            
            #Set current_state to the previous state
            state = next_state
        
    game.kill_game()

    return

for params in grid:
    run_training(params, logdir = logdir)
