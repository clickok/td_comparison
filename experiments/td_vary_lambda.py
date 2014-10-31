#!python3
"""
Code for running TD with varying lambda values, chosen from geometric interval.

Lambda = 1 - (1/2)^t, t=0, 1, 2, ..., 10

Produce plots of learning curve 
"""

import click
import csv
import inspect
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import sys 

# Import parent modules
__currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
__parentdir = os.path.dirname(__currentdir)
sys.path.insert(0,__parentdir) 

import algos
import parameters
from mc_evaluation import first_visit_mc

# Import generic experiment running code
import helper 
import run_algo 

def create_parameter(param_name, *args, **kwargs):
	"""
	Given a name and the values needed to specify a parameter, create a 
	parameter object.
	"""

	param_class = getattr(parameters, param_name)
	param 		= param_class(*args, **kwargs)

	return param 


def run_td_experiment(data, algo_params):
	"""
	Run the experiment for the TD algorithm, and return the results.

	Returns
	-------
	delta_lst : list of float 

	theta_lst : list of numpy.ndarray
	"""

	# Determine number of features for the algorithm from the data file
	num_features = len(data[0][1])

	algo = algos.TD(**algo_params)

	delta_lst = []
	theta_lst = []

	# Get the theta values at the timestep
	theta = algo.theta 

	for i, x in enumerate(data[:-1]):
		# Unpack the run data
		obs, fvec, act, reward 	= x
		fvec_p 					= data[i+1][1] 
		
		# Perform an update of the algorithm
		delta = algo.update(fvec, act, reward, fvec_p)

		# Get the theta values at the timestep
		theta = algo.theta.copy()

		# Record the information
		delta_lst.append(delta)
		theta_lst.append(theta)

	return delta_lst, theta_lst

def approx_err_for_weights(theta_lst, fvec_lst, true_theta):
	""" 
	Calculate the RMSE of the approximation for each weight vector in 
	`theta_lst`, given the true weights, for the feature vectors of interest. 
	"""
	err_lst = []
	for i, theta in enumerate(theta_lst):
		approx_values 	= np.array([np.dot(theta, fvec) for fvec in fvec_lst])
		true_values 	= np.array([np.dot(true_theta, fvec) for fvec in fvec_lst])
		err 			= np.sqrt(np.mean((approx_values - true_values)**2))

		err_lst.append(err)

	return err_lst 

def approx_value_err(theta_lst, obs_lst, fvec_dct, value_dct):
	"""
	Calculate the RMSE of the approximation made by each weight vector in 
	`theta_lst`, for the observations of interest, given a mapping between
	the observations and the feature vectors, as well as the true values. 
	
	Parameters
	----------
	theta_lst : list of numpy.ndarray 

	obs_lst : list
		A list of observations (or 'true' states)

	fvec_dct : dict 
		A dictionary mapping each observation to a feature vector

	value_dct : dict 
		A dictionary mapping each observation to its true value.
	
	Returns 
	-------
	err_lst : 
		A list of the RMSE for each weight vector in theta_lst 
	"""
	err_lst = []
	for i, theta in enumerate(theta_lsta):
		total_err = 0
		for obs in obs_lst:
			fvec = fvec_dct[obs]
			true_val = value_dct[obs]
			approx_val = np.dot(fvec, theta)
			err = (approx_val - true_val)**2
			total_err += err 
		err_lst.append(np.sqrt(np.mean(total_err)))

	return err_lst


# # Plot per-episode, averaging over all episodes
# if __name__ == "__main__" and True:
# 	# data_sources = helper.gen_find('./data/', '1000eps_RandomWalk-17-states_RandomBinomial-n-7-k-3_RandomAction_*.csv')
# 	data_sources = helper.gen_find('./data/', '1000eps_RandomWalk-17-states_IntToVector-n-17_RandomAction_*.csv')


# 	lm_values = [1 - (1/2)**i for i in range(0, 11)]
	
# 	lm_values = [0] # REMOVE

# 	for data_path in data_sources:
# 		print("Running experiment on file:", data_path)

# 		# Get list of observed feature vectors for this dataset
# 		data = helper.load_data(data_path)
# 		fvec_lst = helper.get_run_features(data)
# 		fmapping = helper.get_run_feature_mapping(data)
# 		num_episodes = len([x[0] for x in data if x[0] == -1])

# 		print('Number of episodes:', num_episodes)

# 		# Iterate over the various lambda values for the experiment
# 		for lm in lm_values:
# 			print("Using lambda=", lm)

# 			# Set up the parameters for the experiment
# 			num_features = len(data[0][1])
# 			lm_param = create_parameter('Constant', lm)

# 			# algo_params = \
# 			# {
# 			# 	'n' 	: num_features,
# 			# 	'alpha' : create_parameter('Constant', 0.1/num_features),
# 			# 	'gamma' : create_parameter('Constant', 1),
# 			# 	'lmbda': create_parameter('Constant', lm)
# 			# }

# 			algo_params = \
# 			{
# 				'n' 	: num_features,
# 				'alpha' : create_parameter('Constant', 0.1/num_features),
# 				'gamma' : create_parameter('Constant', 1),
# 				'lmbda': create_parameter('Constant', lm)
# 			}

# 			# Run the experiment and get the data of interest
# 			ret = run_td_experiment(data_path, algo_params)
# 			delta_lst = ret[0]
# 			theta_lst = ret[1]

# 			# Get theta at end of each episode
# 			ep_end_theta_lst = [x for i, x in enumerate(theta_lst) if data[i][0]==-1]

# 			# print(theta_lst[-1])
# 			# for k, v in fmapping.items():
# 			# 	print(k, np.dot(theta_lst[-1], v))
# 			err_lst   = approx_err_for_weights(ep_end_theta_lst, fvec_lst, theta_lst[-1])

# 			###################################################################
# 			# Plot a graph
# 			fig, ax = plt.subplots()
# 			ax.plot(err_lst)

# 			fig.gca().set_position((0.1, 0.3, 0.8, 0.6))
# 			ax.set_xlabel("Episode")
# 			ax.set_ylabel("MSE")
# 			plt.show()
# 		print()



# Plot per-episode
if __name__ == "__main__" and True:
	# data_sources = helper.gen_find('./data/', '1000eps_RandomWalk-17-states_RandomBinomial-n-7-k-3_RandomAction_*.csv')
	
	data_dir = "../data/"
	run_dir = "1000eps_RandomWalk-17-states_IntToVector-n-17"
	data_sources = helper.gen_find(os.path.join(data_dir, run_dir), "*.csv")

	lm_values = [1 - (1/2)**i for i in range(0, 11)]
	lm_values = [0] # REMOVE

	for data_path in data_sources:
		print("Running experiment on file:", data_path)

		# Get list of observed feature vectors for this dataset
		data = helper.load_data(data_path)
		
		#REMOVE######################
		#data = data + data + data
		#############################
		fvec_lst = helper.get_run_features(data)
		fmapping = helper.get_run_feature_mapping(data)
		num_features = len(data[0][1])

		# Iterate over the various lambda values for the experiment
		for lm in lm_values:
			print("Using lambda=", lm)

			gamma_val = 1
			algo_params = \
			{
				'n' 	: num_features,
				'alpha' : create_parameter('Constant', 0.1/num_features),
				'gamma' : create_parameter('Constant', gamma_val),
				'lmbda': create_parameter('Constant', lm),
				'I': create_parameter('Heaviside', 1, 1)
			}

			# Run the experiment and get the data of interest
			ret = run_td_experiment(data, algo_params)
			delta_lst = ret[0]
			theta_lst = ret[1]

			# Determine "optimal" weights
			avg_ret   = first_visit_mc(data, gamma_val)
			# Determine Phi matrix
			fvec_mat 	= np.array([k for k in avg_ret])
			ret_mat 	= np.array([avg_ret[k] for k in avg_ret])
			# Use least squares regression to determine what weights should look like
			v_star, res, rank, singular = np.linalg.lstsq(fvec_mat, ret_mat)

			print("Phi Matrix")
			print(fvec_mat)
			print("Average Return Vector")
			print(ret_mat)
			print("Calculated 'V*'")
			print(v_star)

			# Get theta at end of each episode
			ep_end_theta_lst = [x for i, x in enumerate(theta_lst) if data[i][0]==-1]

			# print(theta_lst[-1])
			# for k, v in fmapping.items():
			# 	print(k, np.dot(theta_lst[-1], v))
			err_lst   = approx_err_for_weights(ep_end_theta_lst, fvec_lst, v_star)

			###################################################################
			# Plot a graph
			fig, ax = plt.subplots()
			ax.plot(err_lst)

			fig.gca().set_position((0.1, 0.3, 0.8, 0.6))
			ax.set_xlabel("Episode")
			ax.set_ylabel("MSE")
			plt.show()
		print()


# Plot per-timestep
if __name__ == "__main__" and False:
	# data_sources = helper.gen_find('./data/', '1000eps_RandomWalk-17-states_RandomBinomial-n-7-k-3_RandomAction_*.csv')
	data_sources = helper.gen_find('./data/1000eps_RandomWalk-17-states_IntToVector-n-17', '1000eps_RandomWalk-17-states_IntToVector-n-17_RandomAction_*.csv')


	lm_values = [1 - (1/2)**i for i in range(0, 11)]
	
	lm_values = [0] # REMOVE

	for data_path in data_sources:
		print("Running experiment on file:", data_path)

		# Get list of observed feature vectors for this dataset
		data = helper.load_data(data_path)
		fvec_lst = helper.get_run_features(data)
		fmapping = helper.get_run_feature_mapping(data)

		# Iterate over the various lambda values for the experiment
		for lm in lm_values:
			print("Using lambda=", lm)

			# Set up the parameters for the experiment
			num_features = len(data[0][1])
			lm_param = create_parameter('Constant', lm)

			# algo_params = \
			# {
			# 	'n' 	: num_features,
			# 	'alpha' : create_parameter('Constant', 0.1/num_features),
			# 	'gamma' : create_parameter('Constant', 1),
			# 	'lmbda': create_parameter('Constant', lm)
			# }

			algo_params = \
			{
				'n' 	: num_features,
				'alpha' : create_parameter('Constant', 0.1/num_features),
				'gamma' : create_parameter('Constant', 1),
				'lmbda': create_parameter('Constant', lm)
			}

			# Run the experiment and get the data of interest
			ret = run_td_experiment(data_path, algo_params)
			delta_lst = ret[0]
			theta_lst = ret[1]

			# print(theta_lst[-1])
			# for k, v in fmapping.items():
			# 	print(k, np.dot(theta_lst[-1], v))
			err_lst   = approx_err_for_weights(theta_lst, fvec_lst, theta_lst[-1])

			###################################################################
			# Plot a graph
			fig, ax = plt.subplots()
			ax.plot(err_lst)

			fig.gca().set_position((0.1, 0.3, 0.8, 0.6))
			ax.set_xlabel("Timestep")
			ax.set_ylabel("MSE")
			plt.show()
		print()

			

