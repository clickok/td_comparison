#!python3
"""
Implementing features for reinforcement learning agents. 

A given feature object is used by the agent to convert the observations from 
interacting with the environment into a feature vector on which it performs 
learning. 

Feature objects may encode memory of past observations, and therefore have 
state.   

Observations in the terminal state are defined to be `-1`, and so the features 
for the terminal state should therefore be entirely zero as well.
"""

import numpy as np

from abc import ABCMeta 


class Feature(metaclass=ABCMeta):
	_name = "Feature" 
	def __init__(self, *args, **kwargs):
		pass 

	def consider(self, obs):
		""" 
		Determine what the new feature *would* be, given an observation, but 
		WITHOUT updating internal state.
		"""
		pass

	def update(self, obs):
		""" 
		Determine what the new feature vector is from the observation, and 
		update internally if necessary.
		"""
		pass  

	def __call__(self, obs):
		"""
		Generate a new feature vector, given an observation. 
		"""
		pass 

	def info_string(self):
		return self._name

class FeatureVector(Feature):
	"""
	Base class for feature functions which must return a numpy array.
	"""
	# TODO: Determine how to ensure return value is a vector/numpy array
	# Perhaps as a super() method with a decorator in init
	def __init__(self, *args, **kwargs):
		pass

	def __call__(self, obs):
		pass 

	def consider(self, obs):
		pass

	def update(self, obs):
		pass 


class IntToVector(FeatureVector):
	"""
	Given an integer observation, converts it to a vector with a single 
	nonzero entry. 
	"""
	_name = "IntToVector" 
	def __init__(self, n=None, *args, **kwargs):
		self.n = n
		self.array = np.arange(self.n)

	def __call__(self, obs):
		if obs == -1:
			ret = np.zeros(self.n)
		else:
			ret = np.zeros(self.n)
			ix  = obs
			ret[ix] = 1
		return ret 

	def info_string(self):
		return "{}-n-{}".format(self._name, self.n)


class RandomBinomial(FeatureVector):
	"""
	A random mapping between integer states and binary vectors of length `n`,
	which each have `k` nonzero bits.

	Uses Python's builtin dictionary to associate hashable keys (which 
	correspond to observations/states) with a random vector with the 
	aforementioned characteristics. 
	"""
	_name = "RandomBinomial"
	def __init__(self, n=None, k=None, *args, **kwargs):
		self.n = n
		self.k = k
		self.mapping = {-1: np.zeros(n)}

	def generate(self):
		""" Generate a new (random) binary vector. """
		ret = np.zeros(self.n)
		ix  = np.random.choice(np.arange(self.n), self.k, replace=False)
		ret[ix] = 1 
		return ret 

	def __call__(self, obs):
		""" 
		Return the feature associated with `val` in the mapping, or if it is 
		not present, generate it. 
		"""
		if obs in self.mapping:
			return self.mapping[obs]
		else:
			self.mapping[obs] = self.generate()
			return self.mapping[obs]

	def info_string(self):
		return "{}-n-{}-k-{}".format(self._name, self.n, self.k)
 