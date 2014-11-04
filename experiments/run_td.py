#!python3
"""
Script to run the TD learning algorithm on datasets matching a pattern.
"""

import click
import fnmatch
import inspect
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
import os  
import random 
import sys 
import yaml 

# Import generic experiment running code
import helper 
import run_algo 

# Import parent modules
__currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
__parentdir = os.path.dirname(__currentdir)
sys.path.insert(0,__parentdir) 

import algos
import parameters

def create_parameter(param_name, *args, **kwargs):
	"""
	Given a name and the values needed to specify a parameter, create a 
	parameter object.
	"""
	param_class = getattr(parameters, param_name)
	param 		= param_class(*args, **kwargs)
	return param 

@click.command()
@click.argument('directory')
@click.argument('pattern')
@click.option('--alpha', default=0.1)
@click.option('--gamma', default=1.0)
@click.option('--lmbda', default=0.0)
def run_td_exp(directory, pattern, alpha, gamma, lmbda):
	""" 
	Run TD algorithm on files in given directory, w/ specified parameters.
	"""
	data_sources = helper.gen_find(os.path.abspath(directory), pattern)
	for data_path in data_sources:
		print(data_path)
		# Load the data
		data  = yaml.load(open(data_path, "r"))
		step_lst = data['steps']
		valid_states = data['valid_states']
		true_values = data['state_values']

		fvec0 = step_lst[0][1]
		num_features = len(fvec0)
		print("Number of features:", num_features)

		obs_set  = helper.get_run_observations(step_lst)		
		obs_map  = helper.get_run_feature_mapping(step_lst)

		print("Running experiment...")
		theta_lst = run_td(step_lst, num_features, alpha, gamma, lmbda)
		print("Total weight updates recorded:", len(theta_lst))

		final_theta = theta_lst[-1]

		print("Final theta values:")
		print(final_theta)

		diff_lst = []
		for obs, fvec in obs_map.items():
			fvec = np.array(fvec)
			true_val = true_values[obs]
			approx_val = np.dot(final_theta, fvec)
			print(obs, fvec, approx_val, true_val)
			diff = approx_val - true_val
			diff_lst.append(diff)

		print(diff_lst)
		diff_lst = np.array(diff_lst)
		print("RMSE:", np.sqrt(np.mean(diff_lst**2)))



def compute_true_theta(step_lst, true_values):
	""" Compute the true theta values """
	obs_set  = helper.get_run_observations(step_lst)		
	obs_map  = helper.get_run_feature_mapping(step_lst)
	fvec_set = helper.get_run_features(step_lst)	

	# Determine the ideal theta, based on how many times states occur
	true_theta = np.zeros_like(final_theta)
	obs_seen   = {x:0 for x in obs_set}
	fvec_seen  = {tuple(x):0 for x in fvec_set}
	for x in step_lst:
		obs = x[0]
		fvec = tuple(x[1])
		obs_seen[obs] += 1
		fvec_seen[fvec] += 1

	# Add based on how many times an observation was seen
	for key, val in obs_map.items():
		true_theta += true_values[key] * np.array(val) * obs_seen[key]/fvec_seen[tuple(val)]

	return true_theta




def run_td(step_lst, num_feat, alpha, gamma, lmbda):
	""" 
	Run TD algorithm on data given by `steps`, return list of weights at each
	timestep. 
	"""
	algo_params = \
	{
		'n'		: num_feat,
		'alpha' : create_parameter('Constant', alpha),
		'gamma' : create_parameter('Constant', gamma),
		'lmbda' : create_parameter('Constant', lmbda),
	}
	A = algos.TD(**algo_params)

	# Set up data structures
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

	return theta_lst


if __name__ == "__main__":
	run_td_exp()