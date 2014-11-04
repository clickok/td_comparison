#!python3
"""
A quick check of the various algorithms.
"""

import click
import numpy as np 
import sys

import algos 
import environment
import features
import policy
import parameters

def clear_terminal():
	CSI="\x1B["
	print(CSI+"1J", end="")

def move_cursor_back(n):
	""" Move the cursor back `n` lines. """
	CSI="\x1B["
	print(CSI+str(n)+"A")

@click.command()
@click.argument('check')
def run_check(check):
	if check == 'td':
		check_td()
	elif check == 'toetd':
		check_toetd()
	elif check == 'lstd':
		check_lstd()
	elif check == 'elstd':
		check_elstd()
	else:
		print("unrecognized option", check)

def check_td():
	# Generate some test data
	num_states = 11
	E 	= environment.RandomWalk(num_states)
	# phi = features.IntToVector(num_states)
	phi = features.RandomBinomial(11, 5)
	pol = policy.RandomAction([-1, 1])

	# Run the experiment for a certain number of episodes
	episode_limit 	= 2000
	episode_count 	= 0
	step_lst 		= []

	while episode_count < episode_limit:
		obs 	= E.observe()
		fvec	= phi(obs)
		act 	= pol(fvec)
		reward, obs_p = E.do(act)

		# Record a step of the episode
		step_lst.append([obs, fvec, act, reward])

		if E.is_terminal():
			episode_count += 1
			fvec_p = phi(obs_p)
			act_p  = pol(fvec_p)

			# Record step
			step_lst.append([obs_p, fvec_p, act_p, 0])

			E.reset()

	# Get some information about the run
	true_values = E.true_state_values()
	obs_set = list(set([x[0] for x in step_lst]))
	fvec_set = list(set(tuple(x[1]) for x in step_lst))
	obs_map = {x[0] : x[1] for x in step_lst}
	fvec_mat = np.array([obs_map[x] for x in sorted(obs_set)])

	# Set up the algorithm
	algo_params = \
	{
		'n' 	: num_states,
		'alpha' : parameters.Constant(0.01),
		'gamma' : parameters.Constant(1.0),
		'lmbda' : parameters.Constant(0.0),
		'print_debug': True,
	}
	A 	= algos.TD(**algo_params)

	# Run the algorithm on the generated data
	theta_lst = []
	for i, step in enumerate(step_lst[:-1]):
		obs, fvec, act, reward = step 
		fvec_p 				   = np.array(step_lst[i+1][1])
		delta = A.update(fvec, act, reward, fvec_p)
		theta = A.theta.copy()
		
		approx_err = [true_values[k] - np.dot(theta, v) for k, v in obs_map.items()]
		rmse       = np.sqrt(np.mean(np.array(approx_err)**2)) 
		# print(obs, reward, delta)


		print(np.array_str(theta, precision=3, suppress_small=True), rmse)
		move_cursor_back(7)

	print()
	
def check_toetd():
	# Generate some test data
	num_states = 11
	E 	= environment.RandomWalk(num_states)
	phi = features.IntToVector(num_states)
	pol = policy.RandomAction([-1, 1])

	# Run the experiment for a certain number of episodes
	episode_limit 	= 2000
	episode_count 	= 0
	step_lst 		= []

	while episode_count < episode_limit:
		obs 	= E.observe()
		fvec	= phi(obs)
		act 	= pol(fvec)
		reward, obs_p = E.do(act)

		# Record a step of the episode
		step_lst.append([obs, fvec, act, reward])

		if E.is_terminal():
			episode_count += 1
			fvec_p = phi(obs_p)
			act_p  = pol(fvec_p)

			# Record step
			step_lst.append([obs_p, fvec_p, act_p, 0])

			E.reset()

	# Get some information about the run
	true_values = E.true_state_values()
	obs_set = list(set([x[0] for x in step_lst]))
	fvec_set = list(set(tuple(x[1]) for x in step_lst))
	obs_map = {x[0] : x[1] for x in step_lst}

	# Set up the algorithm
	algo_params = \
	{
		'n' 	: num_states,
		'alpha' : parameters.Constant(0.05),
		'gamma' : parameters.Constant(1.0),
		'lmbda' : parameters.Constant(0.1),
		'I'		: parameters.Heaviside(1,1),
		'print_debug': True,
	}
	A 	= algos.TOETD(**algo_params)

	# Run the algorithm on the generated data
	theta_lst = []
	for i, step in enumerate(step_lst[:-1]):
		obs, fvec, act, reward = step 
		fvec_p 				   = np.array(step_lst[i+1][1])
		delta = A.update(fvec, act, reward, fvec_p)
		theta = A.theta.copy()
		
		approx_err = [true_values[k] - np.dot(theta, v) for k, v in obs_map.items()]
		rmse       = np.sqrt(np.mean(np.array(approx_err)**2)) 
		# print(obs, reward, delta)

		print(np.array_str(theta, precision=3, suppress_small=True), rmse)
		move_cursor_back(11)
	print()

def check_lstd():
	# Generate some test data
	num_states = 11
	E 	= environment.RandomWalk(num_states)
	phi = features.IntToVector(num_states)
	pol = policy.RandomAction([-1, 1])

	# Run the experiment for a certain number of episodes
	episode_limit 	= 2000
	episode_count 	= 0
	step_lst 		= []

	while episode_count < episode_limit:
		obs 	= E.observe()
		fvec	= phi(obs)
		act 	= pol(fvec)
		reward, obs_p = E.do(act)

		# Record a step of the episode
		step_lst.append([obs, fvec, act, reward])

		if E.is_terminal():
			episode_count += 1
			fvec_p = phi(obs_p)
			act_p  = pol(fvec_p)

			# Record step
			step_lst.append([obs_p, fvec_p, act_p, 0])

			E.reset()

	# Get some information about the run
	true_values = E.true_state_values()
	obs_set = list(set([x[0] for x in step_lst]))
	fvec_set = list(set(tuple(x[1]) for x in step_lst))
	obs_map = {x[0] : x[1] for x in step_lst}

	# Set up the algorithm
	algo_params = \
	{
		'n' 	: num_states,
		'gamma' : parameters.Constant(1.0),
		'lmbda' : parameters.Constant(0.0),
		'epsilon' : 0.001,
		'print_debug': True,
	}
	A 	= algos.LSTD(**algo_params)

	# Run the algorithm on the generated data
	theta_lst = []
	for i, step in enumerate(step_lst[:-1]):
		obs, fvec, act, reward = step 
		fvec_p 				   = np.array(step_lst[i+1][1])
		delta = A.update(fvec, act, reward, fvec_p)
		theta = A.theta.copy()
		
		approx_err = [true_values[k] - np.dot(theta, v) for k, v in obs_map.items()]
		rmse       = np.sqrt(np.mean(np.array(approx_err)**2)) 
		# print(obs, reward, delta)


		print(np.array_str(theta, precision=3, suppress_small=True), rmse)
		move_cursor_back(2)
	print()

def check_elstd():
	# Generate some test data
	num_states = 11
	E 	= environment.RandomWalk(num_states)
	phi = features.IntToVector(num_states)
	pol = policy.RandomAction([-1, 1])

	# Run the experiment for a certain number of episodes
	episode_limit 	= 2000
	episode_count 	= 0
	step_lst 		= []

	while episode_count < episode_limit:
		obs 	= E.observe()
		fvec	= phi(obs)
		act 	= pol(fvec)
		reward, obs_p = E.do(act)

		# Record a step of the episode
		step_lst.append([obs, fvec, act, reward])

		if E.is_terminal():
			episode_count += 1
			fvec_p = phi(obs_p)
			act_p  = pol(fvec_p)

			# Record step
			step_lst.append([obs_p, fvec_p, act_p, 0])

			E.reset()

	# Get some information about the run
	true_values = E.true_state_values()
	obs_set = list(set([x[0] for x in step_lst]))
	fvec_set = list(set(tuple(x[1]) for x in step_lst))
	obs_map = {x[0] : x[1] for x in step_lst}

	# Set up the algorithm
	algo_params = \
	{
		'n' 	: num_states,
		'gamma' : parameters.Constant(1.0),
		'lmbda' : parameters.Constant(0.0),
		'I'		: parameters.Heaviside(1,1),
		'epsilon' : 0.001,
		'print_debug': True,
	}
	A 	= algos.EmphaticLSTD(**algo_params)

	# Run the algorithm on the generated data
	theta_lst = []
	for i, step in enumerate(step_lst[:-1]):
		obs, fvec, act, reward = step 
		fvec_p 				   = np.array(step_lst[i+1][1])
		delta = A.update(fvec, act, reward, fvec_p)
		theta = A.theta.copy()
		
		approx_err = [true_values[k] - np.dot(theta, v) for k, v in obs_map.items()]
		rmse       = np.sqrt(np.mean(np.array(approx_err)**2)) 
		# print(obs, reward, delta)

		print(np.array_str(theta, precision=3, suppress_small=True), rmse)
		move_cursor_back(2)

	print()

if __name__ == "__main__":
	run_check()
