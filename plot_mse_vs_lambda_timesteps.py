#!python3
"""
Code for plotting mean squared error 
"""

import csv
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import sys 

import algos
import parameters


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

def theta_vs_lm_ts(data, algo_name, fixed_params, lm_values, num_ts=50):
	"""
	Set up an algorithm object using `fixed_params` and run it on the dataset
	represented by `data`, while varying `lmbda`. 

	Return a  list of the form [(lm, ts, theta), ...], where `lm` records the 
	value of lambda for that run, and `ts` records the timestep at which the 
	weights were sampled, and `theta` is the weights for that run of the 
	algorithm at that timestep.

	Parameters 
	----------
	data : list of [obs, fvec, action, reward]
		The data on which to run the algorithm and calculate the values. 

	algo_name : str 
		The string used to identify (and therefore find) the algorithm to run. 

	fixed_params : dict 
		The fixed parameters for the algorithms, the same for all runs. 

	lm_values : list of float 
		The lambda values on which to test the algorithm.

	num_ts : int 
		The number of samples to take during each algorithm run, in order to 
		record the evolution of the value function. 

		Each record is (roughly) evenly spaced. 

	Return 
	------
	theta_values : list of [float, int, numpy.ndarray]
		The theta values sampled across multiple lambda values at various times.

	"""

	assert(num_ts >  0)

	# Set up data structures
	theta_values 	= []

	# Determine the timesteps on which to sample the value function
	ts_values 		= np.floor(len(data[:-1]) * np.linspace(0, 1, num_ts+1)[1:])

	# Determine the length of the feature vector
	num_features 		= len(data[0][1])
	fixed_params["n"] 	= num_features

	# Get the algorithm class based on the supplied name
	algo_class 			= getattr(algos, algo_name)


	for lm in lm_values:
		params 	= dict({"lmbda": parameters.Constant(lm)}, **fixed_params)
		algo 	= algo_class(**params)

		for i, x in enumerate(data[:-1]):
			# Unpack the run data
			obs, fvec, act, reward 	= x
			fvec_p 					= data[i+1][1] 
			
			# Record theta values if at appropriate timestep
			if i in ts_values:
				theta_values.append((lm, i, algo.theta))

			# Perform an update of the algorithm
			algo.update(fvec, act, reward, fvec_p)

		# Add final theta_values
		theta_values.append((lm, i, algo.theta))

	return theta_values

def calc_mse(theta, fvec_lst, true_values):
	"""
	Calculate the MSE for a given set of weights and corresponding feature 
	vectors, as compared to the true values obtained some other way.

	Parameters
	----------
	theta : numpy.ndarray 
		The weights for the feature vector, using linear function approximation.

	fvec_lst : list of numpy.ndarray 
		The feature vectors for all the states whose values we approximate. 

	true_values : list or numpy.ndarray 
		The true values for each feature vector in fvec_lst.

	Return 
	------
	error : float
		The mean squared error  

	"""
	approx_values = np.array([np.dot(theta, fvec) for fvec in fvec_lst])
	return np.sqrt(np.mean((approx_values - true_values)**2))



#REMOVE
V_star 		= [i/17 for i in range(17)]
V_star[0]  	= 0
V_star[-1] 	= 0
lmbda_max 	= 1.0


if __name__ == "__main__" and True:
	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader) # skip header
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_lmbdas 	= 10 
	num_ts 		= 10000
	num_features = len(data[0][1])
	fixed 		 = 	{
						"alpha": parameters.Constant(0.1),
						"gamma": parameters.Constant(1),
				   	}
	
	lm_array 	= np.linspace(0, lmbda_max, num_lmbdas)

	algo_name 	= "TD"

	theta_data 	= theta_vs_lm_ts(data, algo_name, fixed, lm_array, num_ts=num_ts)

	# Determine the feature vectors for the given data
	fvec_dct 	= {x[0] : x[1] for x in data}
	fvec_lst 	= [val for key, val in fvec_dct.items()]
	# for key, value in fvec_dct.items():
	# 	print(key, value)
	# print(fvec_lst)

	true_values = np.array([np.dot(V_star, fvec) for fvec in fvec_lst])

	mse_values = [(lm, ts, calc_mse(theta, fvec_lst, true_values)) for lm, ts, theta in theta_data]

	fig, ax = plt.subplots()

	# Gridding the data
	mse_values 	= sorted(mse_values)

	# print(len(theta_data))
	# print(len(mse_values))

	lm_data 	= np.array([x[0] for x in mse_values])[::-1]
	ts_data 	= np.array([x[1] for x in mse_values])
	mse_data 	= np.array([x[2] for x in mse_values])

	# Plot colorbar plot
	# cax = ax.imshow(mse_data.reshape(num_lmbdas, num_ts), interpolation="nearest", cmap="hot")
	# fig.colorbar(cax)	

	ax.scatter(lm_data, ts_data, c=mse_data)

	# Label the axes
	ax.set_xlabel("timestep")
	ax.set_ylabel("lambda")

	# Set ticks on the axes
	ts_ticks = sorted(list(set(ts_data)))
	# ax.set_xticks(ts_ticks)
	lm_ticks = sorted(list(set(lm_data)))
	# ax.set_yticks(lm_ticks)


	plt.show()

	for i, lm in enumerate(lm_ticks):
		for j, ts in enumerate(ts_ticks):
			print(lm, ts, mse_data[i*10 + j])
		print()

	for lm in lm_ticks:
		mse_compare = [i[2] for i in mse_values if i[0]==lm]
		print(lm, np.all(mse_compare == mse_compare[0]))  




if __name__ == "__main__" and False:
	print("TD (Graphing)") 
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader) # skip header
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		# alpha 		= parameters.LogDecay(1.0)
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)

		A = algos.TD(num_features, alpha, gamma, lmbda)

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

	print(alpha(A))

	plt.plot(mse_lst)
	plt.show()