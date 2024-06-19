# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:37:22 2019

@author: Joost Krooshof
"""

from uwbm_functions import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function for the logarithmic equation
def func(a, b, x):
    return a * np.log(x) + b

# Input files
input_csv = 'ep_ts.csv'
catchment_properties = 'ep_neighbourhood.ini'
measure_file = 'ep_measure.ini'

# A list of open water discharge capacities, used for the SDF curve
q_list = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
baseline_q = 16.5

sdf = batch_run_sdf(input_csv, catchment_properties, measure_file, 'SDF Curve.csv', q_list, baseline_q=baseline_q).T

# Obtain the logarithmic equation for the pumping capacity
sdf['Treturn'] = 30/(sdf.index+1)

plt.figure(figsize=(15,8))
df_vars = pd.DataFrame(columns=['q','a','b'])
df_vars['q'] = np.zeros(len(sdf.keys()[1:-1]))

for i, key in enumerate(sdf.keys()[1:-1]):
    x = sdf['Treturn'][0:100].reindex(sdf['Treturn'][0:100].index[::-1]).reset_index(drop=True)
    y = sdf[key][0:100].reindex(sdf[key][0:100].index[::-1]).reset_index(drop=True)
    a, b = np.polyfit(np.log(x), y, 1)
    
    df_vars.loc[i] = [key, a, b]
    
    #plt.plot(x,y)
    plt.plot(x, func(a, b, x), label=key)

plt.xscale('log')
plt.ylim(0,40)
plt.legend()

# Calculate required storage capacity for a set of return periods
req_storage = pd.DataFrame()
req_storage['Treturn'] = [1,2,5,10,20,50,100]
for i, key in enumerate(df_vars['q']):
    req_storage[key] = func(df_vars['a'][i], df_vars['b'][i], req_storage['Treturn'])
req_storage = req_storage.set_index('Treturn')
del req_storage.index.name
req_storage = req_storage.T

# SDF Curve
plt.figure(figsize=(15,12))
for key in req_storage:
    x_stor = req_storage.index.values.astype('int') / 1000 * 9060000 / 86400
    y_dis = req_storage[key] * 0.01 * 9060000 # 0.01 for converting ow_area to total area
    plt.plot(x_stor[y_dis>-0.001], y_dis[y_dis>-0.001], label=key, ms=10, marker='.')
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.8)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Required storage capacity (m3)')
plt.xlabel('Discharge capacity (m3/s)')
plt.legend()