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

parameter_list = [
				#'network_name : {}'.format(network_name),
				#'trainer_name : {}'.format(trainer_name),
				'frame_stacks : {}'.format(frame_stacks),
				'games_to_play : {}'.format(games_to_play),
				'max_length_dataset : {}'.format(max_length_dataset),
				'learning_rate : {}'.format(learning_rate),
				'replay_start_size : {}'.format(replay_start_size),
				'mini_batch_size : {}'.format(mini_batch_size),
				'network_copy : {}'.format(network_copy),
				'epoch_length : {}'.format(epoch_length)
				]

with open('.' + '/settings.txt', 'w+') as f:
	[f.write(i + '\n') for i in parameter_list]