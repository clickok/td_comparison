#!python3
"""
Helper functions for handling data for experiments.

Currently implemented with an eye towards CSV format run data, but this might 
change to YAML depending if this is ultimately non-ideal.
"""

import csv 
import fnmatch
import numpy as np 
import os 
import sys 

def convert(obs, fvec, act, reward):
	"""
	A simple conversion function for dealing with CSV data. 

	Handles the fact that a CSV row might represent the run information as a 
	combination of strings or other python datatypes.

	Parameters
	----------
	obs : convertable to int 

	fvec : str

	act : convertable to int 

	reward : convertable to float 

	Returns 
	-------
	ret : (int, numpy.ndarray, int, float)
	"""
	obs 	= int(obs)

	fvec 	= fvec.strip("[]")
	fvec 	= np.fromstring(fvec, sep=" ")
	fvec 	= fvec.astype(np.int)

	act 	= int(act)

	reward 	= float(reward)
	
	ret 	= obs, fvec, act, reward 
	return ret 

def get_run_features(exp_data):
	"""
	Obtain a list of all feature vectors seen during an experiment.
	"""
	fvec_lst = [x[1] for x in exp_data] 
	return np.array(list(set(tuple(x) for x in fvec_lst))) 

def get_run_feature_mapping(exp_data):
	"""
	Obtain a list of all observations and associated features seen during an 
	experiment. Reliant on the fact that each observation has a unique feature 
	vector, but not the other way around.
	"""
	return {x[0] : x[1] for x in exp_data}
	



# def calc_avg_return(exp_data, gamma=1):
# 	""" 
# 	Determine the average return for each feature vector in a run, using 
# 	first-visit Monte Carlo. 

# 	Unfinished, need to reformulate for when feature vector goes unseen for 
# 	an episode to have valid averaging.

# 	Returns
# 	-------
# 	ret : list of (fvec, tuple)
# 	"""
# 	ep_count = 0
# 	# Get the feature vectors actually seen
# 	fvec_set = set(tuple(x[1]) for x in exp_data)
# 	ret_dct  = {x : 0 for x in fvec_set} 		# Averaged returns
# 	tmp_dct  = {x : [] for x in fvec_set}		# Episode returns
# 	seen_dct = {x : False for x in fvec_set} 	# Feature vectors seen

# 	for i, x in enumerate(exp_data):
# 		# Obtain information for timestep
# 		obs  = x[0]
# 		fvec = tuple(x[1])
# 		rew  = x[2]
# 		seen_dct[fvec] = True

# 		# If in terminal state, reset tmp_dct and average
# 		if obs == -1:
# 			ep_count += 1
# 			for key in ret_dct:
# 				ep_avg 		 = sum(tmp_dct[key])/len(tmp_dct[key]) # BAD
# 				ret_dct[key] = (ret_dct[key]*(ep_count) + ep_avg)/(ep_count)
# 				tmp_dct  	 = {x : [] for x in fvec_set}
# 				seen_dct 	 = {x : 0 for x in fvec_set}

# 	print(ret_dct)
# 	print(ep_count)
		

# def calc_lambda_return(exp_data):
# 	"""

# 	Continuing: 
# 	$$G_{t}^{\lambda} = (1-\lambda)\sum_{n=1}^{\infty}\lmbda^{n-1}G_{t}^{(n)}$$

# 	Episodic:
# 	$$G_{t}^{\lambda} = (1-\lambda)\sum_{n=1}^{T-t-1}\lmbda^{n-1}G_{t}^{(n)} 
# 						+ \lambda^{T-t-1}G_t$$

# 	"""
# 	pass 


def load_data(filename):
	""" Load the data for a run, return as a list. """
	with open(filename, "r") as f: 
		reader = csv.reader(f, delimiter="\n")
		next(reader) # skip header
		data = [convert(*rec) for rec in csv.reader(f, delimiter="\t")]
	return data 

def gen_find(topdir, pattern):
	""" Function that generates files in a directory matching a pattern. """
	for path, dirlist, filelist in os.walk(topdir):
		for name in fnmatch.filter(filelist, pattern):
			yield os.path.join(path, name)