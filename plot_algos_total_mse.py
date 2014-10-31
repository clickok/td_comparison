#!python3
"""
Script for plotting the total MSE for the various algorithms against each other
"""

# REMOVE
lmbda_max = 0.9

import csv
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import sys 

import algos
import parameters

#REMOVE
from algos import *

if __name__ == "__main__" and True:
	print("TOE_LSTD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TOE_LSTD(num_features, gamma, lmbda, interest)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(lm_array, mse_lst, label="TOE-LSTD")


if __name__ == "__main__" and True:
	print("LSTD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)

		A = LSTD(num_features, gamma, lmbda)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(lm_array, mse_lst, label="LSTD")

if __name__ == "__main__" and True:
	print("TOE_TD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TOE_TD(num_features, alpha, gamma, lmbda, interest)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(lm_array, mse_lst, label="TOE-TD")

if __name__ == "__main__" and False:
	print("TD (Graphing)")
	import csv 
	import sys
 

	with open(sys.argv[1], "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader)
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]

	num_features = len(data[0][1])

	lm_array 	= np.linspace(0, lmbda_max)
	mse_lst 	= []
	for lm in lm_array:
		alpha 		= parameters.Constant(0.1)
		gamma		= parameters.Constant(1)
		lmbda 		= parameters.Constant(lm)
		interest 	= parameters.Heaviside(1, 0)

		A = TD(num_features, alpha, gamma, lmbda)

		for i, x in enumerate(data[:-1]):
			_, fvec, act, reward = x
			###################################################################
			# Move reset into the agent itself
			###################################################################
			if _ == -1:
				A.z = fvec 
			fvec_p = data[i+1][1] 
			A.update(fvec, act, reward, fvec_p)

		mse_lst.append(np.sqrt(np.mean((V_star -A.theta)**2)))
		print(lm, np.sqrt(np.mean((V_star -A.theta)**2)))

	import matplotlib.pyplot as plt 
	plt.plot(lm_array, mse_lst, label="TD")
	
# REMOVE
if __name__ == "__main__":
	plt.xlim(0, lmbda_max)
	
	plt.xlabel("Lambda")
	plt.ylabel("MSE")
	plt.legend()
	plt.grid(True)
	plt.show()