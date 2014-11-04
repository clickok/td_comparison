#!python3
"""
A script for generating random walk environments, saving the results as YAML 
files, to facilitate storing additional information.

The ideal version would to be able to specify the type of environment to generate
as a proper command line program, but in the interest of getting things done 
quickly, it instead takes in a configuration file and generates the run based
on that. 

Other features worth having: being able to specify the seeds/sources of 
randomness to enhance reproducibility of experiments generated from this code. 
"""

import click
import numpy as np
import os  
import random 
import sys 
import yaml 

from environment import RandomWalk
import features
import policy

@click.command()
@click.argument("filename")
@click.option("--outdir", default='.', help='directoy in which to save output')
@click.option("--count", default=1, help="number of runs to generate")
@click.option("--nosave", is_flag=True, help="print to console instead of saving")
def gen_from_config(filename, count, outdir, nosave):
	""" Generate run data according to a configuration file. """
	config 		= yaml.load(open(filename, "r"))
	output_dir 	= os.path.abspath(outdir)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	
	print("Generating data from config file:", filename)
	print(config)
	
	# Load the envionment, features, and policy classes
	env_class = RandomWalk
	phi_class = getattr(features, config['phi']['name'])
	pol_class = getattr(policy, config['policy']['name'])

	# Generate a `count` total runs
	for i in range(count):
		# Create data structure to store information from run
		data = {}
		# Instantiate the environment, feature, and policy objects
		E 	= env_class(**config['env'])
		phi = phi_class(**config['phi'])
		pol = pol_class(**config['policy'])

		# Get useful information from environment
		data['state_values'] = E.true_state_values()
		data['valid_states'] = E.valid_states

		# Get useful information from feature vector
		data['fvec_length']  = phi.n 

		# Generate the data from environment, based on phi and policy
		steps = generate_data(E, phi, pol, **config["run"])
		data['steps'] = steps 

		# Determine how many features are active, on average
		fvec_lst 	= [x[1] for x in steps]
		fvec_active	= np.mean(np.sum(fvec_lst, axis=1))
		data['expected_active_features'] = fvec_active 

		print("Run: %d \t Steps: %d" %(i, len(steps)))

		# Capture information about run to name output file meaningfully
		epstring 	= "{}eps".format(config["run"]["episode_limit"])
		envstring 	= E.info_string()
		fvecstring 	= phi.info_string()
		polstring 	= pol.info_string()

		# Write the run data to a CSV file
		output_name = epstring + "_" + envstring + "_" + fvecstring + "_" + polstring + "_" + str(i) + ".yml"

		if nosave:
			print(yaml.dump(data))
		else:
			out_path = os.path.join(output_dir, output_name)
			print("Saving data to:", out_path)
			with open(out_path, "w") as out_file:
				yaml.dump(data, out_file)



def generate_data(env, phi, policy, episode_limit=None, step_limit=None):
	"""
	Generate data for the environment, given a policy, and a function 
	approximator, up to the number of steps or episodes specified. 

	At least one of `episode_limit` or `step_limit` has to be specified. 

	Feature vectors are assumed to be numpy arrays, and are converted to lists 
	prior to being stored as part of a step. 

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
	step_lst : list of list 
		A list of the form [[obs, fvec, act, reward], ...]

	"""
	if episode_limit is None:
		episode_limit = sys.maxsize
	if step_limit is None:
		step_limit = sys.maxsize 

	# Check that at least one of `episode_limit` or `step_limit` has been given
	assert(episode_limit < sys.maxsize or step_limit < sys.maxsize)
	episode_count 	= 0
	step_count 		= 0	

	# Set up data container
	step_lst 		= []
	while (step_count < step_limit and episode_count < episode_limit):
		# Take a single step according to the policy
		obs 		  = env.observe()
		fvec 		  = phi(obs).tolist() # for easier serialization
		act 		  = policy(fvec)
		reward, obs_p = env.do(act)
			
		# Record a step of the episode
		step_lst.append([obs, fvec, act, reward])
		step_count += 1
			
		if env.is_terminal():
			episode_count += 1
			# Account for characteristics of terminal state
			fvec_p 	= phi(obs_p).tolist()
			act_p 	= policy(fvec_p)
				
			# Record terminal state data
			step_lst.append([obs_p, fvec_p, act_p, 0])
			step_count += 1

			# Reset the environment 
			env.reset()

	return step_lst

if __name__ == "__main__":
	gen_from_config()