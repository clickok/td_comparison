#!python3
"""
Policies for reinforcement learning experiments
"""

import random
import numpy as np 

class Policy():
	def __init__(self, *args, **kwargs):
		pass 

	def __call__(self, fvec, *args, **kwargs):
		pass 

class RandomAction(Policy):
	_name = "RandomAction"
	def __init__(self, action_set=None, *args, **kwargs):
		self.action_set = action_set

	def __call__(self, fvec):
		return random.choice(self.action_set)

	def info_string(self):
		return self._name