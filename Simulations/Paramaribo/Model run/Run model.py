# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:49:59 2019

@author: Joost Krooshof
"""

import pandas as pd
import numpy as np
import time
from uwbm_functions import *

# =============================================================================
# Input

# Rainfall & evaporation timeseries
input_csv = 'ep_ts.csv'
# Catchment properties .ini file for the initial parameters
catchment_properties = 'ep_neighbourhood.ini'
# Output file name (.csv gets added to it in the script, so this should be left out here)
output_name = 'Effectiveness measures West NL'
# .csv with the parameters for the measures
measures = pd.read_csv('../input/Parameters measures.csv', index_col=0)
# .csv with the parameters for the neighbourhood types
neighbourhood_pars = pd.read_csv('../input/Parameters neighbourhoods.csv')

# Following measures are expections and are implemented by editing the catchment parameters. No effective depth can be assigned to these. A seperate script is run for these measures.
measures_exception = ['3', '25', '26']
measures_exception_title = ['Adding trees to streetscape', 'Urban forest', 'Permeable pavement system (infiltration)']
# =============================================================================

# Create a list will all measures: '.csv measures' and the exceptions
total_measures = list(measures.index) + measures_exception
total_measures_titles = list(measures['title']) + measures_exception_title

# Create DataFrames to store the results in
df = pd.DataFrame(columns=['id', 'Measure', '5', '10', '20', '30', '40', '50', '100'])
df['id'] = total_measures
df['Measure'] = total_measures_titles

df_runoff = df.copy()
df_Ftot = df.copy()
df_gw = df.copy()
df_evap = df.copy()
df_gw.insert(2, 'Baseline', np.zeros(len(df_gw.index)))
df_evap.insert(2, 'Baseline', np.zeros(len(df_evap.index)))

# The following notes the time, which is used to calculate the remaining runtime of the script
time_counter = 0
max_time_counter = len(neighbourhood_pars) * len(total_measures)
start_time = time.time()

# Loop for each neighbourhood type in the 'Parameters neighbourhoods' file
for n in range(len(neighbourhood_pars)):
    print('\n')
    neighbourhood_id = neighbourhood_pars['id_type'][n]
    
    # Base run: run the model without any measure implemented yet. This is needed to do a comparison between base and measure run.
    print('Currently running Neighbourhood ' + str(n+1) + ' - Base run')
    inputdata = read_inputdata(input_csv)
    dict_param = read_parameters_csv(catchment_properties, str(measures.index[0]), neighbourhood_id, apply_measure=False)
    base_run = running(inputdata, dict_param)
    
    # Calculate amount of paved- and unpaved area, required for the Ftot calculation
    Ap = dict_param['tot_pr_area'] + dict_param['tot_cp_area'] + dict_param['tot_op_area']
    Aup = dict_param['tot_up_area']
    
    # Fraction of runoff from unpaved compared to paved, required for the Ftot calculation.
    # This needs adjustments as the runoff sum difference may not be a good indication for the difference in peak runoff
    sum_Rp = base_run[0]['r_cp_meas'].sum() + base_run[0]['r_cp_swds'].sum() + base_run[0]['r_cp_mss'].sum() + base_run[0]['r_cp_up'].sum()
    sum_Rup = base_run[0]['r_up_ow'].sum() + base_run[0]['r_up_meas'].sum()
    frac_Rup_Rp = sum_Rup / sum_Rp
    
    # Loop for each measure in the 'Parameters measures' file
    for i in measures.index:
        measure_id = str(i)
    
        # Calculate depth measure based on the inflow factor and the effective depth. Currently, the default values for inflow factor are used.
        D_eff = [5, 10, 20, 30, 40, 50, 100]
        inflow_factor = measures['Ain_def'][i]
        D_meas = [x * inflow_factor for x in D_eff]
        
        # Measures with an base storage, such as wet ponds, should have the extra depth added to this base storage.
        if measures['stor_btm_meas_t0'][i] > 0:
            D_meas = [x * inflow_factor + measures['stor_btm_meas_t0'][i] for x in D_eff]
        
        # Determine the storage capacity parameter which needs to be changed depending on the amount of storage layers in the measure.
        # Green roofs (extensive) consists of three layers, however, are an exception where the top layer storage needs to be adjusted instead of the bottom layer storage.
        if int(measures['num_stor_lvl'][i]) == 1:
            varkey = 'storcap_int_meas'
        elif measure_id == '16':
            varkey = 'storcap_top_meas'
        elif int(measures['num_stor_lvl'][i]) > 1:
            varkey = 'storcap_btm_meas'
    
        # Determine the runoff variable which acts as a baseline. This is based on the source of the inflow for the measure: pr, cp, op, up, ow.
        if measures['cp_meas_inflow_area'][i] > 0 and measures['op_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_cp_swds'
            Ami = dict_param['tot_cp_area'] + dict_param['tot_op_area']
        elif measures['pr_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_pr_swds'
            Ami = dict_param['tot_pr_area']
        elif measures['cp_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_cp_swds'
            Ami = dict_param['tot_cp_area']
        elif measures['op_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_op_swds'
            Ami = dict_param['tot_op_area']
        elif measures['up_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_up_ow'
            Ami = dict_param['tot_up_area']
        elif measures['ow_meas_inflow_area'][i] > 0:
            baseline_variable = 'r_ow_swds'
            Ami = dict_param['tot_ow_area']
        
        # Determine variable_to_save for runoff based on the area where the uncontrolled runoff is discharged to
        if measures['surf_runoff_meas_OW'][i] == 1:
            variable_to_save = 'q_meas_ow'
        elif measures['surf_runoff_meas_UZ'][i] == 1:
            variable_to_save = 'q_meas_uz'
        elif measures['surf_runoff_meas_GW'][i] == 1:
            variable_to_save = 'q_meas_gw'
        elif measures['surf_runoff_meas_SWDS'][i] == 1:
            variable_to_save = 'q_meas_swds'
        elif measures['surf_runoff_meas_MSS'][i] == 1:
            variable_to_save = 'q_meas_mss'
        elif measures['surf_runoff_meas_Out'][i] == 1:
            variable_to_save = 'q_meas_out'
        
        # Runoffcap_stor_dependent checks whether the runoff capacity of the measure depends on the storage capacity, e.g. in an underground storage or rain barrel
        if measures.loc[i]['runoffcap_stor_dependent'] == 1:
            runoffcap_factor = measures.loc[i]['runoffcap_stor_factor']
            runoffcap_meas = [x * runoffcap_factor for x in D_meas]
            runoff, gw, evap = run_measures(input_csv, catchment_properties, measure_id, neighbourhood_id, output_name, base_run, varkey = varkey, vararrlist1 = D_meas, correspvarkey='runoffcap_btm_meas', vararrlist2=runoffcap_meas, baseline_variable=baseline_variable, variable_to_save=variable_to_save)
        else:
            runoff, gw, evap = run_measures(input_csv, catchment_properties, measure_id, neighbourhood_id, output_name, base_run, varkey = varkey, vararrlist1 = D_meas, correspvarkey=None, vararrlist2=None, baseline_variable=baseline_variable, variable_to_save=variable_to_save)
        
        num_years = (pd.to_datetime(runoff.Date[len(runoff.Date)-1]) - pd.to_datetime(runoff.Date[0])).days / 365
        num_years = round(num_years)
        
        # 'getconstants_measures' calculates the runoff reduction factors for the measures at each effective depth   
        constants_runoff, mean_constants_runoff = getconstants_measures(runoff, num_year=num_years)
        gw = round(gw / num_years, 2)
        evap = round(evap / num_years, 2)
        
        idx_measure = np.where(df_runoff.id==i)[0][0]
        df_runoff.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = mean_constants_runoff
        df_gw.loc[idx_measure, ['Baseline', '5', '10', '20', '30', '40', '50', '100']] = gw.values[0]
        df_gw.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] -= gw['Baseline'].values[0]
        df_evap.loc[idx_measure, ['Baseline', '5', '10', '20', '30', '40', '50', '100']] = evap.values[0]
        df_evap.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] -= evap['Baseline'].values[0]
        
        # =============================================================================
        #     Calculate Ftot: Runoff reduction factor over the total area
        # =============================================================================
        
        # Calculate the Ftot according to the inflow area of the measure, assuming no measures are taken on unpaved or open water
        Fmeas = mean_constants_runoff
        df_Ftot.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = np.round( ( Ap * np.exp(Ami*np.log(Fmeas)/Ap) + frac_Rup_Rp * Aup ) / ( Ap + frac_Rup_Rp * Aup ), 2)
        
        # Calculate the remaining runtime of the script
        end_time = time.time()
        runtime = end_time - start_time
        time_counter += 1
        fraction = time_counter / max_time_counter
        timeleft = round(((1/fraction) - 1) * runtime / 60,1)
        print(str(timeleft) + ' minutes remaining')
    
    # Loop for the exception measures, which are implemented by altering the catchment itself, rather than implemented as a separate measure.
    for i in measures_exception:
        measure_id = i
        
        if i in ['3','26']:
            baseline_variable = 'r_op_swds'
            variable_to_save = 'r_op_swds'
        elif i == '25':
            baseline_variable = 'r_op_swds'
            variable_to_save = 'r_up_ow'
            
        runoff, gw, evap = running_measure(input_csv, catchment_properties, measure_id, neighbourhood_id, base_run, baseline_variable, variable_to_save)
        
        num_years = (pd.to_datetime(runoff.Date[len(runoff.Date)-1]) - pd.to_datetime(runoff.Date[0])).days / 365
        num_years = round(num_years)
        
        constants_runoff, mean_constants_runoff = getconstants_measures(runoff, num_year=num_years)
        gw = round(gw / num_years, 2)
        evap = round(evap / num_years, 2)
        
        idx_measure = np.where(df_runoff.id==i)[0][0]
        df_runoff.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = mean_constants_runoff
        
        df_gw.loc[idx_measure, 'Baseline'] = gw['Baseline'].values[0]
        df_gw.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = gw['alt'].values[0]
        
        df_evap.loc[idx_measure, 'Baseline'] = evap['Baseline'].values[0]
        df_evap.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = evap['alt'].values[0]        
        
        # Calculate Ftot
        Fmeas = mean_constants_runoff
        Ami = dict_param['tot_op_area']
        df_Ftot.loc[idx_measure, ['5', '10', '20', '30', '40', '50', '100']] = np.round( ( Ap * np.exp(Ami*np.log(Fmeas)/Ap) + frac_Rup_Rp * Aup ) / ( Ap + frac_Rup_Rp * Aup ), 2)

    neighbourhood_name = neighbourhood_pars['title'][n]
    output_file = output_name + ' - %s.xlsx' % neighbourhood_name
    
    # Save the results in a .xsls (Excel workbook) file
    with pd.ExcelWriter('pysol/' + output_file) as writer:
        df_runoff.to_excel(writer, sheet_name='Runoff factor', index=0)
        df_Ftot.to_excel(writer, sheet_name='Ftot', index=0)
        df_gw.to_excel(writer, sheet_name='Groundwater recharge', index=0)
        df_evap.to_excel(writer, sheet_name='Evaporation', index=0)