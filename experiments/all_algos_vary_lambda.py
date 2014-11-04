if __name__ == "__main__":
	""" 
	Perform runs over episodes in a data directory for varying lambda values,
	using the TD-lambda algorithm. 
	
	Average the MSE error at the end of each episode, and plot MSE vs episode 
	number. 
	"""
	data_dir 	 = "./data/randomwalk_tabular/"
	data_pattern = "*.yml"
	data_sources = list(helper.gen_find(os.path.abspath(data_dir), data_pattern))

	graph_dir 	 = os.path.join(__parentdir, 'graphs')

	# lm_values 	 = [1 - (1/2)**i for i in range(2)]
	# lm_values = [1 - (1/2)**i for i in range(0, 11)] + [1]
	lm_values = [0, 0.25, 0.5, 0.75, 1]
	algo_lm_dct	 = {lm:[] for lm in lm_values}

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
				'alpha' : create_parameter('Constant', 0.1/expected_active),
				'gamma' : create_parameter('Constant', 1),
				'lmbda': create_parameter('Constant', lm),
			}

			A = algos.TD(**algo_params)

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

			lm_dct[lm].append(mse_lst)

	# Average the errors at the end of each episode for each value of lambda
	err_max = 0
	for k, v in lm_dct.items():
		assert(all([len(x) == len(v[0]) for x in v])) # ensure equal length
		xdata = np.arange(len(v[0]))
		lm_err_vals = np.array(v)
		ydata = np.mean(lm_err_vals, axis=0)
		err_max  = max(np.max(ydata), err_max)
		plt.plot(xdata, ydata, label="lambda={}".format(k))

	print("Largest error value seen:", err_max)
	plt.xlabel("Episode")
	plt.ylabel("MSE")
	plt.ylim([0, min(2, err_max)])
	plt.legend()
	plt.show()

	dir_name  = os.path.basename(data_dir.strip('/')) 
	save_name = "td_vary_lambda" + dir_name + '.png'
	save_path = os.path.join(graph_dir, save_name)
	plt.savefig(save_path)