# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:19:48 2019

@author: chen_sg
"""
from uwbm_functions import *

# Input files
input_csv = 'xt_test.csv'
catchment_properties = 'ep_neighbourhood.ini'
measure_file = 'ep_measure.ini'

# Run the model. 'base_run' is a pandas DataFrame in which all model variables are stored.
inputdata = read_inputdata(input_csv)
dict_param = read_parameters(catchment_properties, measure_file)
base_run = running(inputdata, dict_param)

