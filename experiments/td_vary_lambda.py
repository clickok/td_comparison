#!python3
"""
Code for running TD with varying lambda values, chosen from geometric interval.

Lambda = 1 - (1/2)^t, t=0, 1, 2, ..., 10

Produce plots of learning curve 
"""

import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import sys 
import yaml

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
		theta_lst.append(theta)

	return theta_lst

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



if __name__ == "__main__":
	data_dir 	 = "../data/randomwalk_tabular"
	data_pattern = "*.yml"
	data_sources = list(helper.gen_find(os.path.abspath(data_dir), data_pattern))

	lm_values 	 = [1 - (1/2)**i for i in range(2)] + [1]
	# lm_values = [1 - (1/2)**i for i in range(0, 11)] + [1]

	for data_path in data_sources:
		print(data_path)
		# Load the data
		data = yaml.load(open(data_path, "r"))
		step_lst = data['steps']
		valid_states = data['valid_states']
		true_values = data['state_values']
		expected_active = data['expected_active_features']
		num_features = len(step_lst[0][1]) 

		obs_lst  = [x[0] for x in step_lst]
		obs_set  = helper.get_run_observations(step_lst)		
		obs_map  = helper.get_run_feature_mapping(step_lst)

		print("Number of steps in data:", len(step_lst))
		print("Number of episodes in data:", len([x for x in obs_lst if x == -1]))


		for lm in lm_values:
			print("Using lambda=", lm)
			algo_params = \
			{
				'n' 	: num_features,
				'alpha' : create_parameter('Constant', 0.1/expected_active),
				'gamma' : create_parameter('Constant', 1),
				'lmbda': create_parameter('Constant', lm),
			}

			A = algos.TD(**algo_params)

			# Store weights during run
			theta_lst = []
			theta_lst.append(A.theta)

			for i, x in enumerate(step_lst[:-1]):
				# Unpack the run data
				obs, fvec, act, reward 	= x
				fvec 					= np.array(fvec)
				fvec_p 					= np.array(step_lst[i+1][1])

				# Perform an update of the algorithm
				delta = A.update(fvec, act, reward, fvec_p)
				theta = A.theta.copy()

				theta_lst.append(theta)





