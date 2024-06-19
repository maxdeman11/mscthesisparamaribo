# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 08:46:55 2019

@author: Joost Krooshof
"""

from uwbm_functions import *

# Input files
input_csv = 'ep_ts.csv'
catchment_properties = 'ep_neighbourhood.ini'
measure_file = 'ep_measure.ini'

# Run the model. 'base_run' is a pandas DataFrame in which all model variables are stored.
inputdata = read_inputdata(input_csv)
dict_param = read_parameters(catchment_properties, measure_file)
base_run = running(inputdata, dict_param)