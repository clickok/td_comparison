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


def run_toetd_experiment(data, algo_params):
	"""
	Run the experiment with given parameters on a dataset, return the results.

	Returns
	-------
	delta_lst : list of float 

	theta_lst : list of numpy.ndarray
	"""
	
	# Determine number of features for the algorithm from the data file
	num_features = len(data[0][1])

	algo = algos.TOETD(**algo_params)

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

def approx_err(theta_lst, fvec_lst, true_theta):
	""" 
	Calculate the MSE of the approximation at each timestep, for the given 
	weights vs. the true weights, for the feature vectors of interest. 
	"""
	err_lst = []
	for i, theta in enumerate(theta_lst):
		approx_values 	= np.array([np.dot(theta, fvec) for fvec in fvec_lst])
		true_values 	= np.array([np.dot(true_theta, fvec) for fvec in fvec_lst])
		err 			= np.sqrt(np.mean((approx_values - true_values)**2))

		err_lst.append(err)

	return err_lst 
