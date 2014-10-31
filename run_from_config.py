#!python3
"""
Script for running algorithms from configuration files.
"""

import csv 
import click 
import numpy as np 
import sys 
import yaml 

import algos 
import parameters 


@click.command()
@click.argument("data_path", help="path to the file containing data")
@click.argument("param_path", help="path ")
def run_from_config(data_path, param_path):
	""" 

	"""
	# Load the data
	with open(data_path, "r") as f:
		reader 	= csv.reader(f, delimiter="\n")
		next(reader) # skip header
		data 	= [convert(*rec) for rec in csv.reader(f, delimiter="\t")]
	
	# Load the parameters
	param_dct = yaml.load(open(param_path, "r"))
	print(param_dct)