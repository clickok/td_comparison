#!python3
"""
Code implementing parameters as objects which are callable on an agent and
determine the resulting parameter based on the internal values of that agent.
"""

import numpy as np 

class Parameter():
	""" The parameter base class. """
	def __init__(self, *args, **kwargs):
		pass 

	def __call__(self, algo, *args, **kwargs):
		""" 
		Given an algorithm which makes the data necessary to compute the 
		parameter's value, perform the calculation and return the result.

		Parameters 
		----------
		algo : Algorithm 
			An algorithm object which has the information needed to compute the 
			parameter's value accessible as attributes or methods.

			For example, we might wish for the learning rate of an algorithm to 
			decay in order to be assured of convergence. Therefore, the 
			algorithm needs to keep track of the current iteration (timestep) 
			so that the parameter can determine how much its initial value 
			should be reduced at the present time.
		"""
		raise NotImplementedError("Parameter abstract base class called")


class Constant():
	_name = "ConstantParameter"
	def __init__(self, value, *args, **kwargs):
		self.value = value 

	def __call__(self, agent, *args, **kwargs):
		return self.value 

class LogDecay():
	""" Decay according to value(t) = value(0)/t """
	_name = "LogDecayParameter"
	def __init__(self, value, *args, **kwargs):
		self.value = value 

	def __call__(self, agent, *args, **kwargs):
		ts = agent.t 
		if ts == 0:
			return self.value 
		return (self.value/ts)

class LinearDecay():
	""" Decay according to $x_t = (rate^t)x_0"""
	_name = "LinearDecayParameter"
	pass 

class QuadraticDecay():
	""" Decay according to $x_t = ((rate^2)^t)x_0.$ """
	_name = "QuadraticDecayParameter"
	pass 

class Heaviside():
	""" Returns `value` until t > `drop`, then returns 0"""
	_name = "HeavisideParameter"
	def __init__(self, value, drop, *args, **kwargs):
		self.value = value 
		self.drop = drop

	def __call__(self, agent, *args, **kwargs):
		ts = agent.t 
		if ts <= self.drop:
			return self.value 
		else:
			return 0 