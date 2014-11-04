#!python3
"""
Helper functions for handling data for experiments.

Currently implemented with an eye towards CSV format run data, but this might 
change to YAML depending if this is ultimately non-ideal.
"""

import fnmatch
import numpy as np 
import os 
import sys 
import yaml

def get_run_observations(step_lst):
	""" Return a list of all observations seen during an experiment. """
	obs_lst = [x[0] for x in step_lst]
	return list(set(obs_lst))

def get_run_features(step_lst):
	"""
	Obtain a list of all feature vectors seen during an experiment.
	"""
	fvec_lst = [x[1] for x in step_lst] 
	return np.array(list(set(tuple(x) for x in fvec_lst))) 

def get_run_feature_mapping(step_lst):
	"""
	Obtain a list of all observations and associated features seen during an 
	experiment. Reliant on the fact that each observation has a unique feature 
	vector, but not the other way around.
	"""
	return {x[0] : x[1] for x in step_lst}
	

def load_data(filename):
	""" Load the data for a run, return as a list. """
	with open(filename, "r") as f: 
		data 	 = yaml.load(f)
		step_lst = data['steps']
	return data 

def gen_find(topdir, pattern):
	""" Function that generates files in a directory matching a pattern. """
	for path, dirlist, filelist in os.walk(topdir):
		for name in fnmatch.filter(filelist, pattern):
			yield os.path.join(path, name)