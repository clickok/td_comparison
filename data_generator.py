#!python3
"""
Generate runs from an environment in order to have a consistent set of data 
from which to test the performance of various algorithms.
"""

import click
import csv 
import numpy as np
import os  
import random 
import sys 
import yaml 

from functools import reduce

import environment 
import features
import policy

@click.command()
@click.argument("filename")
@click.option("--count", default=1, help="number of runs to generate")
def gen_from_config(filename, count):
	""" Generate run data from a configuration file """
	config = yaml.load(open(filename, "r"))
	print("Generating data from config file...")
	print(config)

	for i in range(count):
		# Define the environment
		env_class = getattr(environment, config["env"]["name"])
		E 	= env_class(**config["env"])

		# Define the function mapping observations to features
		phi_class = getattr(features, config["phi"]["name"])
		Phi = phi_class(**config["phi"])

		# Define the policy
		policy_class = getattr(policy, config["policy"]["name"])
		Policy = policy_class(**config["policy"])

		# Actually generate the data
		ret = generate_data(E, Phi, Policy, **config["run"])

		epstring 	= "{}eps".format(config["run"]["episode_limit"])
		envstring 	= E.info_string()
		fvecstring 	= Phi.info_string()
		polstring 	= Policy.info_string()

		# Check that the length of all lists are equal
		assert(all([len(i) == len(ret[0]) for i in ret]))

		# Write the run data to a CSV file
		output_dir 	= os.path.abspath("./data/")
		output_name = epstring + "_" + envstring + "_" + fvecstring + "_" + polstring + "_" + str(i) + ".csv"
		

		print("saving data to: %s" % output_name)
		with open(os.path.join(output_dir, output_name), "w") as out_file:
			writer = csv.writer(out_file, delimiter="\t",)
			writer.writerow(["obs", "fvec", "action", "reward"])
			writer.writerows(zip(*ret))

		print(len(ret[0]))


def generate_data(env, phi, policy, episode_limit=None, step_limit=None):
	"""
	Generate data for the environment, given a policy, and a function 
	approximator, up to the number of steps or episodes specified. 

	At least one of `episode_limit` or `step_limit` has to be specified. 

	Parameters
	----------
	env : Environment
		An environment from which to generate the data. 

	phi : function 
		A function which maps observations to feature vectors.

	policy : function
		A policy function which maps observations to actions.

	episode_limit : int (optional)
		An integer which specifies how many episodes of data to generate. 

	step_limit : int (optional)
		An integer which specifies how many steps of data to generate.
	
	Returns
	-------
	obs_lst : list of observations 

	fvec_lst : list of feature vectors

	act_lst : list of actions  

	reward_lst :  list of rewards 
	"""
	if episode_limit is None:
		episode_limit = sys.maxsize
	if step_limit is None:
		step_limit = sys.maxsize 

	# Check that at least one of `episode_limit` or `step_limit` has been given
	assert(episode_limit < sys.maxsize or step_limit < sys.maxsize)
	episode_count 	= 0
	step_count 		= 0	

	# Set up the data containers
	obs_lst 		= []
	fvec_lst 		= []
	act_lst 		= []
	reward_lst 		= []

	while (step_count < step_limit and episode_count < episode_limit):
			# Take a single step according to the policy
			obs 			= env.observe()
			fvec 			= phi(obs)
			act 			= policy(fvec)
			reward, obs_p 	= env.do(act)
			# Record a step of the episode
			obs_lst.append(obs) 
			fvec_lst.append(fvec)
			act_lst.append(act)
			reward_lst.append(reward)
			
			if env.is_terminal():
				step_count += 1
				fvec 	= phi(obs_p)
				act 	= policy(fvec)
				reward  = 0

				# Record terminal state data
				obs_lst.append(obs_p)
				fvec_lst.append(fvec)
				act_lst.append(act)
				reward_lst.append(reward)
				env.reset()
				episode_count += 1

			step_count += 1

	return obs_lst, fvec_lst, act_lst, reward_lst

if __name__ == "__main__":
	gen_from_config()

# Folded into code allowing runs to be executed from config file
# if __name__ == "__main__" and False:
# 	import features

# 	num_states = 7 
# 	episode_limit = 100

# 	E 	= environment.RandomWalk(num_states)
# 	Fa 	= features.IntToVector(num_states)
# 	Pi 	= lambda x : random.choice([-1, 1])

# 	ret = generate_data(E, Fa, Pi, episode_limit)

# 	obs_lst, fvec_lst, act_lst, reward_lst = ret 

# 	epstring 	= "{}eps".format(episode_limit)
# 	envstring 	= E.info_string()

# 	# Check that the length of all lists are equal
# 	assert(all([len(i) == len(ret[0]) for i in ret]))

# 	with open(epstring + "_" + envstring + ".csv", "w") as out_file:
# 		writer = csv.writer(out_file, delimiter="\t",)
# 		writer.writerow(["obs", "fvec", "action", "reward"])
# 		writer.writerows(zip(*ret))
		
