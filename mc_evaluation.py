#!python3
"""
Script for performing Monte-Carlo estimation of returns for states.

These functions could probably be substantially shortened by applying suitable
list comprehensions...
"""

import csv
import numpy as np 
import os 
import sys 

import experiments.helper as helper 

def first_visit_mc(data, gamma=1):
	"""
	Perform first-visit Monte-Carlo to calculate optimal value function for 
	the states/features appearing in a series of episodes.

	Parameters
	----------
	data : list 
		A list of the form [(obs, fvec, action, reward), ...]

	gamma : float 
		The discounting parameter 

	Returns 
	-------
	avg_ret : dict 
		A dictionary with keys representing feature vectors and values 
		corresponding to the return associated with those features, averaged 
		over the episodes in `data`. 
	"""
	fv_length = len(data[0][1])

	# List of feature vectors & the unique feature vectors
	fvec_lst 	= [tuple(x[1]) for x in data]
	fvec_set 	= set(tuple(x) for x in fvec_lst)

	ret_dct 	= {x : [] for x in fvec_set}
	ep_dct 		= {x : [] for x in fvec_set}
	seen_dct 	= {x : False for x in fvec_set}

	ep_count = 0

	# Go through episodes, determining what rewards followed each state
	for i, x in enumerate(data[:-1]):
		obs, fvec, act, reward 	= x
		fvec_p 					= data[i+1][1] 

		# Convert feature vector to tuple
		fv 						= tuple(fvec)
		seen_dct[fv] 			= True 

		# Append reward to features that have been observed
		for k in fvec_set:
			if seen_dct[k]:
				ep_dct[k].append(reward)


		# If in terminal state, reset
		if obs == -1:
			ep_count += 1 
			for x in fvec_set:
				ret_dct[x].append(ep_dct[x])
			ep_dct 		= {x : [] for x in fvec_set}
			seen_dct 	= {x : False for x in fvec_set}

	# Print information to check that everything is running correctly
	print("Total episodes seen:", ep_count)

	# Determine discounted sums for each feature in each episode
	ret_sum = {x : [] for x in fvec_set} 
	for fv in fvec_set:
		for chain in ret_dct[fv]:
			if len(chain) == 0: # if feature unseen, don't consider it
				continue
			total = 0
			for i, x in enumerate(chain):
				total += x * (gamma)**i 
			ret_sum[fv].append(total)

	# Average the returns
	avg_ret = {x : 0 for x in fvec_set}
	for fv in fvec_set:
		if len(ret_sum[fv]) > 0:
			avg_ret[fv] = sum(ret_sum[fv])/len(ret_sum[fv])

	return avg_ret 

def first_visit_mc_weights(data, gamma=1):
	"""
	Perform first-visit Monte-Carlo to calculate optimal value function for 
	the states/features appearing in a series of episodes.

	Parameters
	----------
	data : list 
		A list of the form [(obs, fvec, action, reward), ...]

	gamma : float 
		The discounting parameter 

	Returns 
	-------
	theta_star : numpy.ndarray
		The "optimal" weights, as calculated by First-Visit Monte Carlo

	NB: Assumes that feature vectors are {0,1}-valued 
	"""
	fv_length 	= len(data[0][1])

	# List of feature vectors & the unique feature vectors
	fvec_lst 	= [tuple(x[1]) for x in data]
	fvec_set 	= set(tuple(x) for x in fvec_lst)

	ret_dct 	= {x : [] for x in fvec_set}
	ep_dct 		= {x : [] for x in fvec_set}
	seen_dct 	= {x : False for x in fvec_set}

	ep_count = 0

	# Go through episodes, determining what rewards followed each state
	for i, x in enumerate(data[:-1]):
		obs, fvec, act, reward 	= x
		fvec_p 					= data[i+1][1] 

		# Convert feature vector to tuple
		fv 						= tuple(fvec)
		seen_dct[fv] 			= True 

		# Append reward to features that have been observed
		for k in fvec_set:
			if seen_dct[k]:
				ep_dct[k].append(reward)


		# If in terminal state, reset
		if obs == -1:
			ep_count += 1 
			for x in fvec_set:
				ret_dct[x].append(ep_dct[x])
			ep_dct 		= {x : [] for x in fvec_set}
			seen_dct 	= {x : False for x in fvec_set}

	# Print information to check that everything is running correctly
	print("Total episodes seen:", ep_count)
	ep_count = 0

	# Determine discounted sums for each feature in each episode
	ret_sum = {x : [] for x in fvec_set} 
	for fv in fvec_set:
		for chain in ret_dct[fv]:
			if len(chain) == 0: # if feature unseen, don't consider it
				continue
			total = 0
			for i, x in enumerate(chain):
				total += x * (gamma)**i 
			ret_sum[fv].append(total)

	# Average the returns
	avg_ret = {x : 0 for x in fvec_set}
	for fv in fvec_set:
		if len(ret_sum[fv]) > 0:
			avg_ret[fv] = sum(ret_sum[fv])/len(ret_sum[fv])

	# Determine Phi matrix
	fvec_mat 	= np.array([k for k in avg_ret])
	ret_mat 	= np.array([avg_ret[k] for k in avg_ret])

	# Use least squares regression to determine ideal weights
	theta_star, res, rank, singular = np.linalg.lstsq(fvec_mat, ret_mat)

	return theta_star

# TESTING
if __name__ == "__main__":
	gamma = 1
	data_dir = "./data/"
	# run_dir = "1000eps_RandomWalk-17-states_IntToVector-n-17"
	run_dir = "1000eps_RandomWalk-17-states_RandomBinomial-n-7-k-3"

	data_sources = helper.gen_find(os.path.join(data_dir, run_dir), "*.csv")

	for data_path in data_sources:
		data = helper.load_data(data_path)

		# Use first visit monte carlo functions
		avg_ret = first_visit_mc(data, 1)
		theta_star = first_visit_mc_weights(data, 1)

		## Check if the functions perform as expected ##
		fvec_lst 	= [tuple(x[1]) for x in data]
		fvec_set 	= set(tuple(x) for x in fvec_lst)
		obs_map 	= {x[0]: tuple(x[1]) for x in data}
		
		# Construct map from feature-->observations
		fv_map = {}
		for key, value in obs_map.items():
			if value not in fv_map:
				fv_map[value] = [key]
			else:
				fv_map[value].append(key) 

		print("Estimated values: (fvec, obs, avg_ret, fvec * theta_star)")
		for k in sorted(list(fvec_set), reverse=True):
			print(k, fv_map[k], avg_ret[k], np.dot(theta_star, k))

		print("Differences between estimates")
		for k in sorted(list(fvec_set), reverse=True):
			print(k, abs(avg_ret[k]- np.dot(theta_star, k)))

		# HARD CODING
		print("Difference between approximation and 'actual' values")
		for k in sorted(list(fvec_set), reverse=True):
			print(k, abs(np.array(fv_map[k])/16 - np.dot(theta_star, k)))		

		print("Percent difference between approximation and 'actual' values")
		for k in sorted(list(fvec_set), reverse=True):
			actual = np.array(fv_map[k])/16
			print(k, 100 * abs((actual - np.dot(theta_star, k))/actual))


		print("Approximation RMSE:")
		err_lst = [np.dot(theta_star, obs_map[k]) - k/16 for k in obs_map]
		mse 	= np.sqrt(np.mean(np.power(err_lst, 2)))
		print(mse)