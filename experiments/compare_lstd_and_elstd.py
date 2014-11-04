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
	# data_dir 	 = "./data/randomwalk_randombinomial/"
	data_dir 	 = './data/randomwalk_tabular'
	data_pattern = "*.yml"
	data_sources = list(helper.gen_find(os.path.abspath(data_dir), data_pattern))

	graph_dir 	 = os.path.join(__parentdir, 'graphs')

	# Specify parameters
	alpha0 = 0.1

	# lm_values 	 = [1 - (1/2)**i for i in range(2)]
	# lm_values = [1 - (1/2)**i for i in range(0, 11)] + [1]
	lm_values = [0, 0.25, 0.5, 0.75, 1]
	emph_lm_dct  = {lm:[] for lm in lm_values}
	plain_lm_dct  = {lm:[] for lm in lm_values}

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
			plain_lstd_algo_params = \
			{
				'n' 	: num_features,
				'gamma' : create_parameter('Constant', 1),
				'lmbda' : create_parameter('Constant', lm),
				'epsilon' : 1e-6
			}

			emph_lstd_algo_params = \
			{
				'n' 	: num_features,
				'gamma' : create_parameter('Constant', 1),
				'lmbda' : create_parameter('Constant', lm),
				'I'		: create_parameter('Heaviside', 1, 1),
				'epsilon' : 1e-6
			}

			# Instantiate algorithms
			emph_lstd  = algos.EmphaticLSTD(**emph_lstd_algo_params)
			plain_lstd = algos.LSTD(**plain_lstd_algo_params)

			# Store weights during run for each algorithm
			plain_mse_lst   = []
			plain_theta_lst = []
			plain_theta_lst.append(plain_lstd.theta)

			emph_mse_lst   = []
			emph_theta_lst = []
			emph_theta_lst.append(emph_lstd.theta)
			for i, x in enumerate(step_lst[:-1]):
				# Unpack the run data
				obs, fvec, act, reward 	= x
				fvec 					= np.array(fvec)
				fvec_p 					= np.array(step_lst[i+1][1])

				# Perform an update of the algorithms
				plain_delta = plain_lstd.update(fvec, act, reward, fvec_p)
				plain_theta = plain_lstd.theta.copy()

				emph_delta = emph_lstd.update(fvec, act, reward, fvec_p)
				emph_theta = emph_lstd.theta.copy()
				# Append theta if at the end of an episode
				if step_lst[i+1][0] == -1:
					plain_theta_lst.append(plain_theta)
					approx_err = [true_values[k] - np.dot(plain_theta, v) for k, v in obs_map.items()]
					plain_mse 	   = np.mean(np.array(approx_err)**2) 	
					plain_mse_lst.append(plain_mse)

					emph_theta_lst.append(emph_theta)
					approx_err = [true_values[k] - np.dot(emph_theta, v) for k, v in obs_map.items()]
					emph_mse 	   = np.mean(np.array(approx_err)**2) 	
					emph_mse_lst.append(emph_mse)


			print("MSE:", np.mean(plain_mse_lst), np.mean(emph_mse_lst))
			plain_lm_dct[lm].append(plain_mse_lst)
			emph_lm_dct[lm].append(emph_mse_lst)

	# Plot MSE vs lambda for each algorithm
	xdata 			= sorted([x for x in lm_values])
	plain_ydata 	= []
	emph_ydata		= []

	# Average the errors at the end of each episode for each value of lambda
	err_max = 0
	for lm in xdata:
		plain_err = np.mean(plain_lm_dct[lm])
		emph_err  = np.mean(emph_lm_dct[lm])

		plain_ydata.append(plain_err)
		emph_ydata.append(emph_err)
	
	plt.plot(xdata, emph_ydata, label="ELSTD")
	plt.plot(xdata, plain_ydata, label="LSTD")

	print("Largest error value seen:", err_max)
	plt.xlabel("Lambda")
	plt.ylabel("MSE")
	plt.ylim([0, min(2, err_max)])
	plt.legend()

	plt.show()

	# dir_name  = os.path.basename(data_dir.strip('/')) 
	# save_name = "elstd_vary_lambda" + dir_name + '.png'
	# save_path = os.path.join(graph_dir, save_name)
	# plt.savefig(save_path)
