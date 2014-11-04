#!python3
"""
Environments in which to test reinforcement learning algorithms.

Terminal states are defined as having state value `-1`

The only action available to an agent in the terminal state are all `-1`
"""

import numpy as np 
import random

class Environment():
	"""
	A single-agent synchronous environment (MDP) base class. 

	Attributes
	----------

	num_states : int 

	valid_actions : list of actions 

	_name : str 
	""" 
	def __init__(self, *args, **kwargs):
		pass 

	def do(self, action):
		""" 
		Update the environment according to an agent's action, returning a new
		state and a reward.
		"""
		pass 

	def observe(self, state=None):
		""" 
		Return the observation associated with the given state, or if the state 
		is not specified, return the observation associated with the current 
		state of the environment.
		"""
		if state is None:
			state = self._state 

		if self.is_terminal(state):
			return - 1
		else: 
			return state

	def is_terminal(self):
		pass

	def reset(self):
		pass 

	@property
	def valid_actions(self):
		return self._valid_actions
	
	@valid_actions.setter
	def valid_actions(self, value):
		self._valid_actions = value

	@property
	def valid_states(self):
		return self._valid_states
	
	@valid_states.setter
	def valid_states(self, value):
		self._valid_states = value

	def valid_transitions(self, state):
		""" 
		Compute all possible transitions from given state for the environment. 
		"""
		pass 
		 

	def reward(self, s, a, sp):
		""" 
		Return the reward for transitioning to state `sp` from state `s` having
		taken action `a`. 
		"""
		pass 

	def print_rewards(self):
		"""
		Print the rewards for all possible state-action-state transitions. 
		"""
		pass 

	def true_state_values(self, gamma=1):
		""" 
		Return a dictionary of the form {obs:value, ...} reflecting the true 
		values of each state in the environment, subject to discounting.
		"""
		raise NotImplementedError

	def info_string(self):
		""" Return short string containing information about the environment"""
		return self._name 


class RandomWalk(Environment):
	"""
	A n-state random walk environment. 
	(cf. Sutton & Barto, 1998, Chapter 6)

	A simple, discrete line environment, where the agent can transition from 
	a given state to the one on the "left" or the "right", at random. The agent 
	starts in the middle of the environment, with the leftmost and rightmost 
	states being terminal states. Tranisitioning to the leftmost state 
	(state `0`) has a reward of 0, while transitioning to the rightmost state 
	(state `n-1`) has a reward of 1.   
	"""
	_valid_actions 	= [1, -1]
	_name 			= "RandomWalk"
	def __init__(self, num_states=None, *args, **kwargs):
		#super().__init__(self, num_states)
		self.num_states = num_states
		self._state 	= num_states // 2
		self._valid_states = [i for i in range(1, num_states-1)]
		self._valid_states.append(-1)

	def info_string(self):
		return "{}-{}-states".format(self._name, self.num_states) 

	def do(self, action):
		"""
		Parameters
		----------
		action : int  

		Returns
		-------
		reward : float 

		new_obs : int 
		"""
		# Here we ignore the action, because random walk.
		if self.is_terminal():
			raise Exception("Cannot perform action in terminal state.")
		s  = self._state
		a  = random.choice(self.valid_actions)
		sp = s + a 

		# Update agent position and return reward
		self._state = sp 
		return (self.reward(s, a, sp), self.observe())

	def reward(self, state, action, new_state):
		if new_state == self.num_states-1:
			return 1 
		else:
			return 0

	def is_terminal(self, state=None):
		# If state not passed, use current state of environment
		if state is None:
			state = self._state
		
		if state in (0, self.num_states-1):
			return True 
		else:
			return False 

	def reset(self, new_state=None):
		if new_state is None:
			self._state = self.num_states // 2
		else:
			self._state = new_state

	@property
	def valid_states(self):
		return self._valid_states

	def true_state_values(self):
		# TODO: reimplement as matrices to handle `gamma` parameter
		tmp 	= self.num_states-1
		val_dct = {i:i/tmp for i in range(1, tmp)}
		val_dct[-1] = 0
		return val_dct

class Treelike(Environment):
	"""
	A single-agent synchronous environment, with 'tree-like' structure. 

	The 'leaves' of the tree (of which there are $b^l$, where b is the 
	branching factor and l is the number of layers) are the states prior to the 
	terminal state; only transitions from leaf to terminal state have non-zero
	value. 

	Parameters
	----------
	layers : int 
		The number of non-leaf layers in the tree. 
	branches : int
		The number of child branches each node has. This also represents how 
		many actions are available to the agent at each time step for 
		non-terminal nodes. 
	p : float, optional
		A parameter specifying the amount of influence or predictability that
		the agent has on the environment. Specifically, (1-p) is the 
		probability that the state transition will be random. For example, p=1
		means the environment is completely predictable; conversely p=0 means
		the environment is essentially random.   

		By default, the state transitions are set to random. 
	
	Attributes
	----------
	_state : int 
		The agent's position in the environment. Represented as an integer.
		
		The root will always be state 0, and leaf nodes will be in the interval
		$[(b**l -1)/(b - 1), (b**(l+1) - 1)/(b - 1)$, which can be seen from 
		the geometric series expression for the number of states. 

		To determine which state corresponds to node `x` in layer `L`, we use 
		the formula $s = (\sum_{i=0}^L b^i) + x - 1$, which is pretty ugly, but 
		could also be expressed as $s = (b^L - 1)/(b - 1) + x - 1$, with `x` in 
		$\{0, 1, 2, ..., b-1\}$. 

		A terminal state will always be *observed* to be -1, however. 
	"""
	_name = "Treelike"
	def __init__(self, branches=None, layers=None, p=0, *args, **kwargs):
		# TODO: Add super() if needed
		self.branches 		= branches
		self.layers 		= layers
		self.p 				= p 
		self.valid_actions 	= [i for i in range(1, branches+1)]
		self.num_states 	= int((branches**(layers + 1) - 1) / (branches - 1))
		self.num_leaves 	= int(branches**layers)

		# Initialize the values of leaf nodes to standard normal
		self.state_rewards = np.zeros(self.num_states)	
		self.state_rewards[-self.num_leaves:] = np.random.randn(self.num_leaves)	

		# Set the agent's initial position to the root of the tree 
		self._state = 0


	def reward(self, state, action, new_state):
		""" Reward is a function of the next state only. """
		return self.state_rewards[new_state] 

	def reset(self, new_state=0):
		""" 
		Reset the environment, setting the agent at random (the default) or at 
		a specified point-- either a tuple or an integer.
		""" 
		self._state = new_state

	def do(self, action):
		"""
		Perform action in current state, returning the result.

		We can represent states as either a 2-tuple or an integer. It is 
		possible to translate the state tuple (l, x) to the state integer n 
		via $n = (b^l - 1) + x$. 

		If in state $s_t = n$, the possible next states are integers in the 
		interval $[n*b + 1, n*b + b]$. 

		Parameters
		----------
		action : int 
			A valid action is an integer between 1 and `b`.

		Returns
		-------
		reward : float 

		new_obs : int 
		"""
		assert(1 <= action <= self.branches)
		if random.random() > self.p:
			action = np.random.choice(self.valid_actions)
		
		# Update state and calculate reward
		sp 			= (self.branches * self._state) + action 
		reward 		= self.reward(self._state, action, sp)
		self._state = sp 

		return (reward, self.observe())

	def is_terminal(self, state=None):
		""" 
		Check if the given state is terminal, or if a state is not specified,
		determine if the current state is terminal (ie, one of the leaf nodes).
		"""
		# If state not passed, use current state of environment
		if state is None:
			state = self._state

		if (self.num_states - self.num_leaves) <= state < self.num_states: 
			return True 
		else: 
			return False 

	def info_string(self):
		return "{}-b-{}-l-{}-p-{}".format(self._name, self.branches, self.layers, self.p) 

class Tube(Environment):
	"""
	A environment that resembles a lattice on a 3D cylinder.

	Parameters
	----------
	length : int 

	width : int 

	p : float, optional
	"""

	_name = "Tube"
	def __init__(self, length=None, width=None, p=0, *args, **kwargs):
		self.length 		= length
		self.width 			= width
		self.p 				= p 
		self.valid_actions 	= [-1, 0, 1] 
		self.num_states 	= length * width 
		self.end_states 	= (length - 1) * width

		self._state 		= 0

	def do(self, action):
		"""
		Perform action in current state, returning the result.

		We can represent states as either a 2-tuple or an integer. It is 
		possible to translate the state tuple (l, x) to the state integer n 
		via $n = (b^l - 1) + x$. 

		If in state $s_t = n$, the possible next states are integers in the 
		interval $[n*b + 1, n*b + b]$. 

		Parameters
		----------
		action : int 
			A valid action is an integer between 1 and `b`.

		Returns
		-------
		reward : float 

		new_obs : int 
		"""
		pass

	def reward(self, state, action, new_state):
		pass 

	def is_terminal(self):
		pass 

	def reset(self, new_state=0):
		""" 
		Reset the environment, setting the agent at random (the default) or at 
		a specified point-- either a tuple or an integer.
		""" 
		self._state = 0 

	def info_string(self):
		return "{}-l-{}-w-{}-p-{}".format(self._name, self.length, self.width, self.p) 

class Hypercube(Environment):
	"""
	A lattice-like environment, which can be imagined as a plane or a cube.

	The "edges" of the plane are terminal states, with one edge having +1 
	and the other having 0 reward.
	"""
	_name = "Hypercube"
	def __init__(self, shape, *args, **kwargs):
		#super().__init__(self, shape, *args, **kwargs)
		# TODO: Add code to ensure proper setup
		# Store relevant variables
		self.shape = shape

		# Set agent initial position
		self.state = tuple([i//2 for i in shape])

	def do(self, action):
		""" 
		Update the environment according to an agent's action, returning a new
		state and a reward.
		"""
		pass 

	def valid_actions(self):
		# TODO: New name
		# It is tempting to make this a static method, but probably not worth.
		# However, you could do interesting things w/ decorators in that regard...
		pass 

	def valid_transitions(self):
		# TODO: New name
		# It is tempting to make this a static method, but probably not worth.
		pass 

	def state_num(self):
		# If the MDP has a countable number of states, then is it worth 
		# implementing an isomorphism for each problem to ensure consistent
		# ways of representing the MDPs?
		pass 

	def get_state(self):
		# Make this a property thingy?
		pass 

class HyperCylinder(Environment):
	"""
	An environment with the topology of a lattice on an n-dimensional cylinder.

	Essentially, the first dimension has the structure of a finite ring, while 
	the other dimensions are countable sets.  

	For 1-D, it's just a ring, while for 2D it's like an actual cylinder. 
	"""
	pass 

class Tube(HyperCylinder):
	"""
	A environment that resembles a lattice on a 3D cylinder.
	"""
	pass 