import gym
import numpy as np
from colorsys import rgb_to_yiq
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
print(env.action_space)
obs_max = np.max(env.observation_space.high)

def luminecence(observation, obs_max):
	observation = observation.astype('float32')
	observation *= 1.0 / obs_max

	lin_vec = np.array([0.299, 0.587, 0.114])
	observation = np.tensordot(observation, lin_vec, axes = 1)

	return observation

for i_episode in range(1):
	observation = env.reset()
	for t in range(1000):
		env.render()
		
		action = env.action_space.sample()
		
		observation, reward, done, info = env.step(action)
		observation = luminecence(observation, obs_max)

		print(np.array(observation).shape)
		#print(observation[:,:,0].shape)
		print(np.max(observation))

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break