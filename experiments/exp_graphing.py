#!python3
"""
Functions for plotting graphs of experiments.
"""

def plot_experiment():
	"""
	Effects
	-------
	Plot of TD-error vs. episodes

	Plot of TD-error vs. steps taken
	"""
	pass 

def plot_theta_err_vs_ts(theta_lst, fvec_lst, true_theta):
	""" Plot the MSE of the approximation vs the true values. """

	ts_lst 	= []
	err_lst = []
	for i, theta in enumerate(theta_lst):
		approx_values 	= np.array([np.dot(theta, fvec) for fvec in fvec_lst])
		true_values 	= np.array([np.dot(true_theta, fvec) for fvec in fvec_lst])
		err 			= np.sqrt(np.mean((approx_values - true_values)**2))

		ts_lst.append(i)
		err_lst.append(err)

	fig, ax = plt.subplots()

	return fig, ax 