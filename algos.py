#!python3
"""
Algorithms for reinforcement learning experiments.
"""

import numpy as np 

import parameters

class Algorithm():
	"""
	The base class for an RL algorithm.

	As Currently Implemented: 
	=========================

	The paramters are not constant numerical values such as floats or ints, but
	instead as objects which are called prior to executing the `_update` method.

	The `_update` method contains the internal update rule, which is specific
	to each algorithm.

	In contrast the `update` method is essentially similar in form for all
	algorithms, taking the feature vector for the current observation, the
	action taken in response to this feature vector, the reward associated with
	the transition, and the feature vector for the next state. 

	Attributes
	----------
	theta : numpy.ndarray 
		The weights used to approximate the value function for the feature 
		vector specific to the MDP. 
	"""
	def __init__(self, *args, **kwargs):
		"""
		Initialize the algorithm. 

		Paramters can either be specified in order or via a dictionary 
		containing objects of the required types. 
		"""
		pass 

	def _update(self, fvec, act, reward, fvec_p, *args, **kwargs):
		"""
		Perform an update for the algorithm.

		delta : float  
			The TD-error for the update
		"""
		raise NotImplementedError("_update() called in Algorithm abstract base class")

	def update(self, fvec, act, reward, fvec_p):
		""" 
		Perform an update for the algorithm (by calling the `_update` method) 

		Returns
		-------
		delta : float  
			The TD-error for the update 
		"""
		raise NotImplementedError("update() called in Algorithm abstract base class") 

class TD(Algorithm):
	""" Temporal difference (lambda) algorithm. """
	_name = "TD-Lambda"
	def __init__(self, n=None, alpha=None, gamma=None, lmbda=None, *args, **kwargs):		
		# Set up variables
		self.alpha 	= alpha
		self.gamma 	= gamma
		self.lmbda 	= lmbda 

		# Set up data structures
		self.t 		= 0
		self.n 		= n 
		self.theta 	= np.zeros(n)
		self.z 		= np.zeros(n)

	def _update(self, fvec, act, reward, fvec_p, alpha, gamma, lmbda):
		# TODO: add importance sampling
		# print(fvec)
		# print(self.z)
		# print(self.theta)
		# print(fvec.shape)
		# print(act)
		# print(reward)
		# print(fvec_p.shape)
		# print(alpha)
		# print(gamma)
		# print(lmbda)

		delta 	= reward + np.dot(fvec_p * gamma - fvec, self.theta)
		self.z 	= (lmbda * self.z) + fvec
		self.theta += alpha * delta * self.z 

		return delta 

	def update(self, fvec, act, reward, fvec_p):
		alpha 	= self.alpha(self)
		gamma 	= self.gamma(self)
		lmbda 	= self.lmbda(self)

		self.t += 1

		# Call internal update method
		return self._update(fvec, act, reward, fvec_p, alpha, gamma, lmbda)

class TOETD(Algorithm):
	""" True-Online Emphatic TD-Lambda Algorithm. """
	#TODO: Add in importance sampling
	_name = "TOETD"
	def __init__(self, n=None, alpha=None, gamma=None, lmbda=None, I=None):
		# Initialize the parameters
		self.t 			= 0
		self.n 			= n 

		# Set up variables
		self.alpha 		= alpha
		self.gamma 		= gamma
		self.lmbda 		= lmbda 
		self.I 			= I

		# Set up data structures
		self.old_theta 	= np.zeros(n)
		self.theta 		= np.zeros(n)
		self.z 			= np.zeros(n)
		self.H 			= 0
		self.M 			= self.I(self)
		self.oldI 		= self.I(self)

	def _update(self, fvec, act, reward, fvec_p, alpha, gm, gm_p, lm, lm_p, I_p):
		# print("\n", self.t)
		# print(fvec)
		# print(act)
		# print(reward)
		# print(fvec_p)
		# print(alpha)
		# print(gm)
		# print(gm_p)
		# print(lm)
		# print(lm_p)
		# print(I_p)
		# print(self.z)
		# print(self.M)
		# print(self.H)


		delta 	= reward + np.dot(self.theta, gm_p*fvec_p - fvec)
		self.z 	= gm*lm*self.z + alpha*self.M*(1 - gm*lm*np.dot(self.z, fvec))*fvec 
		theta  	= self.theta + delta*self.z + np.dot(self.theta - self.old_theta, fvec) * (self.z - alpha*self.M*fvec) 
		self.H 	= gm_p*(self.H + self.oldI)
		self.M 	= I_p + (1 - lm_p)*self.H  

		# Update values for the next iteration
		self.oldI = I_p 
		self.old_theta = self.theta
		self.theta = theta 

		return delta

	def update(self, fvec, act, reward, fvec_p):
		alpha 	= self.alpha(self) 
		gm 		= self.gamma(self)
		lm 		= self.lmbda(self)

		self.t += 1
		gm_p	= self.gamma(self)
		lm_p 	= self.lmbda(self)
		I_p		= self.I(self)

		return self._update(fvec, act, reward, fvec_p, alpha, gm, gm_p, lm, lm_p, I_p)

class EmphaticLSTD(Algorithm):
	_name = "EmphaticLSTD"
	def __init__(self, n=None, gamma=None, lmbda=None, interest=None, epsilon=0):
		# Initialize the parameters
		self.t 			= 0
		self.n 			= n 

		# Set up variables
		self.gamma 		= gamma
		self.lmbda 		= lmbda 
		self.I 			= interest

		# Initialize data structures
		self.z 			= np.zeros(n)
		self.A 			= np.eye(n, n) * epsilon
		self.b 			= np.zeros(n)
		self.H 			= 0
		self.M 			= self.I(self)
		self.oldI 		= self.I(self)

	@property 
	def theta(self):
		_theta = np.dot(np.linalg.pinv(self.A), self.b)
		return _theta

	def _update(self, fvec, act, reward, fvec_p, gm, gm_p, lm, lm_p, I_p):
		# print("\n", self.t)
		# print(fvec)
		# print(act)
		# print(reward)
		# print(fvec_p)
		# print(alpha)
		# print(gm)
		# print(gm_p)
		# print(lm)
		# print(lm_p)
		# print(I_p)
		# print(self.z)
		# print(self.M)
		# print(self.H)

		# TODO: ensure this is correct
		self.z 	= gm*lm*self.z + self.M*(1 - gm*lm*np.dot(self.z, fvec))*fvec 
		self.A += np.outer(self.z, (fvec - gm*fvec_p))
		self.b += self.z * reward 	

		self.H 	= gm_p * (self.H + self.oldI)
		self.M 	= I_p + (1 - lm_p)
		
		# Update values for next iteration
		self.oldI = I_p 

	def update(self, fvec, act, reward, fvec_p):
		# Determine variables for current iteration
		gm 		= self.gamma(self)
		lm 		= self.lmbda(self)

		self.t += 1
		gm_p	= self.gamma(self)
		lm_p 	= self.lmbda(self)
		I_p		= self.I(self)

		return self._update(fvec, act, reward, fvec_p, gm, gm_p, lm, lm_p, I_p)


class LSTD(Algorithm):
	""" 
	Implemented according to Boyan's LSTD paper. 

	Parameters 
	----------
	n : the length of the feature vector

	gamma : Parameter 

	lmbda : Parameter 

	epsilon : float 
		Determines the degree of regularization, by setting the initial `A` 
		matrix to the identity matrix multiplied by `epsilon`.
	"""
	_name = "LSTD-Lambda"
	def __init__(self, n=None, gamma=None, lmbda=None, epsilon=0):

		# Store the supplied parameters
		self.n 			= n 
		self.gamma 		= gamma
		self.lmbda 		= lmbda 
		self.epsilon	= epsilon

		# Set up data structures
		self.t 			= 0
		self.z 			= np.zeros(n)
		self.A 			= np.eye(n, n) * epsilon
		self.b 			= np.zeros(n)
		
	@property
	def theta(self):
		_theta = np.dot(np.linalg.pinv(self.A), self.b)
		return _theta

	def _update(self, fvec, act, reward, fvec_p, gamma, lmbda):
		# TODO: Consider adding stepsize to ensure convergence?
		self.z 	= gamma * lmbda * self.z + fvec
		self.A += np.outer(self.z, (fvec - gamma*fvec_p))
		self.b += self.z * reward 

	def update(self, fvec, act, reward, fvec_p):
		gamma 	= self.gamma(self)
		lmbda 	= self.lmbda(self)

		self.t += 1
		self._update(fvec, act, reward, fvec_p, gamma, lmbda)




def convert(obs, fvec, act, reward):
	"""A simple conversion function for dealing with CSV data. """
	obs 	= int(obs)

	fvec 	= fvec.strip("[]")
	fvec 	= np.fromstring(fvec, sep=" ")
	fvec 	= fvec.astype(np.int)

	act 	= int(act)

	reward 	= float(reward)
	
	ret 	= obs, fvec, act, reward 
	return ret 

# REMOVE
V_star 		= [i/17 for i in range(17)]
V_star[0]  	= 0
V_star[-1] 	= 0

if __name__ == "__main__" and False:
	print("TOE_LSTD")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	alpha 		= parameters.Constant(0.1)
	gamma		= parameters.Constant(1)
	lmbda 		= parameters.Constant(0)
	interest 	= parameters.Heaviside(1, 0)

	num_features = len(data[0][1])

	A = TOE_LSTD(num_features, gamma, lmbda, interest)

	for i, x in enumerate(data[:-1]):
		_, fvec, act, reward = x
		fvec_p = data[i+1][1] 
		A.update(fvec, act, reward, fvec_p)

	print(A.theta)
	print(np.sqrt(np.mean((V_star -A.theta)**2))) # MSE

if __name__ == "__main__" and False:
	print("TOETD")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	alpha 		= parameters.Constant(0.1)
	gamma		= parameters.Constant(1)
	lmbda 		= parameters.Constant(0.01)
	interest 	= parameters.Heaviside(1, 0)

	num_features = len(data[0][1])

	A = TOETD(num_features, alpha, gamma, lmbda, interest)

	for i, x in enumerate(data[:-1]):
		_, fvec, act, reward = x
		fvec_p = data[i+1][1] 
		A.update(fvec, act, reward, fvec_p)

	print(A.theta)
	print(np.sqrt(np.mean((V_star -A.theta)**2))) # MSE


if __name__ == "__main__" and False:
	print("LSTD")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	alpha = parameters.Constant(0.1)
	gamma = parameters.Constant(1)
	lmbda = parameters.Constant(0)

	num_features = len(data[0][1])

	A = LSTD(num_features, gamma, lmbda)

	for i, x in enumerate(data[:-1]):
		_, fvec, act, reward = x
		fvec_p = data[i+1][1] 
		A.update(fvec, act, reward, fvec_p)

	print(A.theta)
	print(np.sqrt(np.mean((V_star -A.theta)**2))) # MSE

if __name__ == "__main__" and False:
	print("TD")
	
	import csv 
	import sys

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	alpha = parameters.Constant(0.1)
	gamma = parameters.Constant(1)
	lmbda = parameters.Constant(0)

	num_features = len(data[0][1])

	A = TD(num_features, alpha, gamma, lmbda)

	for i, x in enumerate(data[:-1]):
		_, fvec, act, reward = x
		fvec_p = data[i+1][1] 
		A.update(fvec, act, reward, fvec_p)

	print(A.theta)
	print(np.sqrt(np.mean((V_star -A.theta)**2))) # MSE


# REMOVE
lmbda_max = 0.9

if __name__ == "__main__" and True:
	print("TOE_LSTD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TOE_LSTD(num_features, gamma, lmbda, interest)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(mse_lst)



if __name__ == "__main__" and True:
	print("LSTD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)

		A = LSTD(num_features, gamma, lmbda)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(mse_lst)

if __name__ == "__main__" and True:
	print("TOETD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TOETD(num_features, alpha, gamma, lmbda, interest)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(mse_lst)

if __name__ == "__main__" and False:
	print("TD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TD(num_features, alpha, gamma, lmbda)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(mse_lst)
	
# REMOVE
if __name__ == "__main__":
	plt.show()