#!python3
"""
A script for running agents on test data. 
"""

import click 
import csv 
import fnmatch
import numpy as np 
import os 
import sys 



def run_algo(data, algo_name, algo_params):
	"""
	Run the algorithm on the specified data. 
	Records the theta values during each step of the run. 

	Returns
	-------
	theta_lst : list of numpy.ndarray
		The theta values of each step during the run. 
	"""

	theta_lst = []

	algo_class = getattr(algos, algo_name)

	algo = algo_class(**algo_params)

	for i, x in enumerate(data[:-1]):
		# Unpack the run data 
		obs, fvec, act, reward 	= x 
		fvec_p 					= data[i+1][1]

		algo.update(fvec, act, reward, fvec_p)

		#print(i, algo.theta)
		theta_lst.append(algo.theta)

	return theta_lst 

def gen_find(pattern, topdir):
	""" Function that generates files in a directory matching a pattern. """
	for path, dirlist, filelist in os.walk(topdir):
		for name in fnmatch.filter(filelist, pattern):
			yield os.path.join(path, name)




if __name__ == "__main__" and False:
	# Load the data
	data = load_data(sys.argv[1])

	# Emphatic LSTD Parameters
	alpha = create_parameter('Constant', 0.1)
	gamma = create_parameter('Constant', 1)
	lmbda = create_parameter('Constant', 0)
	interest = create_parameter('Heaviside', 1, 1)
	epsilon = 0.1

	num_features = len(data[0][1])

	algo_params = \
	{
		"n": num_features,
		"gamma" : gamma,
		"lmbda" : lmbda,
		"interest": interest,
		"epsilon": epsilon,
	}


	
	# theta_lst = run_algo(data, "EmphaticLSTD", algo_params)

