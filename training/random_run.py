"""
	A baseline run using random actions
	in the mock gym.
"""
from utils import mock_gym
import numpy as np

env = mock_gym.make()
observation = env.reset()
steps = 10
total_reward = 0
for _ in range(steps):
	action = observation.target
	observation, reward, done, info = env.step(action)
	print reward
	total_reward += reward
	if done:
		print "Done"
		break

