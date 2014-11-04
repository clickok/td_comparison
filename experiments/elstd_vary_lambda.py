#!python3
"""
Code for running elstd with varying lambda values, chosen from geometric interval.

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


if __name__ == "__main__":
	""" 
	Perform runs over episodes in a data directory for varying lambda values,
	using the elstd algorithm. 
	
	Average the MSE error at the end of each episode, and plot MSE vs episode 
	number. 
	"""
	data_dir 	 = "./data/randomwalk_randombinomial/"
	# data_dir 	 = './data/randomwalk_tabular'
	data_pattern = "*.yml"
	data_sources = list(helper.gen_find(os.path.abspath(data_dir), data_pattern))

	graph_dir 	 = os.path.join(__parentdir, 'graphs')

	# Specify parameters
	alpha0 = 0.1

	# lm_values 	 = [1 - (1/2)**i for i in range(2)]
	# lm_values = [1 - (1/2)**i for i in range(0, 11)] + [1]
	lm_values = [0, 0.25, 0.5, 0.75, 1]
	lm_dct 		 = {lm:[] for lm in lm_values}

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

		# Print some information
		print("Number of steps in data:", len(step_lst))
		print("Number of episodes in data:", len([x for x in obs_lst if x == -1]))
		print("Expected number of active features:", expected_active)


		for lm in lm_values:
			print("Using lambda=", lm)
			algo_params = \
			{
				'n' 	: num_features,
				'gamma' : create_parameter('Constant', 1),
				'lmbda' : create_parameter('Constant', lm),
				'I'		: create_parameter('Heaviside', 1, 1),
				'epsilon' : 1e-6
			}

			A = algos.EmphaticLSTD(**algo_params)

			# Store weights during run
			mse_lst   = []
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

				# Append theta if at the end of an episode
				if step_lst[i+1][0] == -1:
					theta_lst.append(theta)
					approx_err = [true_values[k] - np.dot(theta, v) for k, v in obs_map.items()]
					mse 	   = np.mean(np.array(approx_err)**2) 	
					mse_lst.append(mse)

			print("MSE:", np.mean(mse_lst))
			lm_dct[lm].append(mse_lst)

	# Average the errors at the end of each episode for each value of lambda
	err_max = 0
	for k, v in lm_dct.items():
		assert(all([len(x) == len(v[0]) for x in v])) # ensure equal length
		xdata = np.arange(len(v[0]))
		lm_err_vals = np.array(v)
		ydata = np.mean(lm_err_vals, axis=0)
		err_max  = max(np.max(ydata), err_max)
		print("Largest error for lambda=", k, ":", np.max(ydata))
		plt.plot(xdata, ydata, label="lambda={}".format(k))

	print("Largest error value seen:", err_max)
	plt.xlabel("Episode")
	plt.ylabel("MSE")
	plt.ylim([0, min(1, err_max)])
	plt.legend()

	dir_name  = os.path.basename(data_dir.strip('/')) 
	save_name = "elstd_vary_lambda" + dir_name + '.png'
	save_path = os.path.join(graph_dir, save_name)
	plt.savefig(save_path)
