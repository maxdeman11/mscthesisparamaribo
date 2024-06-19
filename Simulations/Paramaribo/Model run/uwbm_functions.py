#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import fire
import logging
import toml
import math
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from tqdm import trange
from time import sleep
from urbanwb.pavedroof import PavedRoof
from urbanwb.closedpaved import ClosedPaved
from urbanwb.openpaved import OpenPaved
from urbanwb.unpaved import Unpaved
from urbanwb.unsaturatedzone import UnsaturatedZone
from urbanwb.groundwater import Groundwater
from urbanwb.sewersystem import SewerSystem
from urbanwb.openwater import OpenWater
from urbanwb.measure import Measure
from urbanwb.selector import soil_selector
from urbanwb.gwlcalculator import gwlcalc
from urbanwb.read_parameter_base import read_parameter_base
from urbanwb.read_parameter_measure import read_parameter_measure
from urbanwb.waterbalance_checker import water_balance_checker
from urbanwb.setlogger import setuplog
from urbanwb.sdf_curve import SDF_curve2, get_segment_index

class UrbanwbModel(object):
    """
    Creates an instance of UrbanwbModel class which consists of all eight components namely paved roof,  closed paved,
    open paved, unpaved, unsaturated zone, groundwater, sewer system and open water together with measure module.
    Iterates __next__() as time stepping to get solutions for all time steps.

    Args:
        dict_param (dictionary): A dictionary of necessary parameters read from neighbourhood config file and measure
        config file to initialize a model instance
    """

    def __init__(self, dict_param):
        self.param = dict_param
        self.pavedroof = PavedRoof(**self.param)
        self.closedpaved = ClosedPaved(**self.param)
        self.openpaved = OpenPaved(**self.param)
        self.unpaved = Unpaved(**self.param)
        self.unsaturatedzone = UnsaturatedZone(
            theta_uz_t0=soil_selector(self.param["soiltype"], self.param["croptype"])[
                gwlcalc(self.param["gwl_t0"])[2]
            ]["moist_cont_eq_rz[mm]"],
            **self.param
        )
        self.groundwater = Groundwater(**self.param)
        self.sewersystem = SewerSystem(**self.param)
        self.openwater = OpenWater(**self.param)
        self.measure = Measure(k_sat_uz=self.unsaturatedzone.k_sat_uz, **self.param)

    def __iter__(self):
        return self

    def __next__(
        self,
        p_atm,
        e_pot_ow,
        ref_grass,
        lst_prevt,
    ):
        """
        Calculates storage, fluxes, coefficients and other related results at current time step.
        """
        try:
            pr_sol = self.pavedroof.sol(p_atm=p_atm, e_pot_ow=e_pot_ow)
            cp_sol = self.closedpaved.sol(p_atm=p_atm, e_pot_ow=e_pot_ow)
            op_sol = self.openpaved.sol(
                p_atm=p_atm, e_pot_ow=e_pot_ow, delta_t=self.param["delta_t"]
            )
            up_sol = self.unpaved.sol(
                p_atm=p_atm,
                e_pot_ow=e_pot_ow,
                r_pr_up=pr_sol["r_pr_up"],
                r_cp_up=cp_sol["r_cp_up"],
                r_op_up=op_sol["r_op_up"],
                theta_uz_prevt=lst_prevt["theta_uz"],
                pr_no_meas_area=self.param["pr_no_meas_area"],
                cp_no_meas_area=self.param["cp_no_meas_area"],
                op_no_meas_area=self.param["op_no_meas_area"],
                ow_no_meas_area=self.param["ow_no_meas_area"],
                delta_t=self.param["delta_t"],
            )
            meas_sol = self.measure.sol(p_atm=p_atm,
                                        e_pot_ow=e_pot_ow,
                                        r_pr_meas=pr_sol["r_pr_meas"], r_cp_meas=cp_sol["r_cp_meas"],
                                        r_op_meas=op_sol["r_op_meas"], r_up_meas=up_sol["r_up_meas"],
                                        pr_no_meas_area=self.param["pr_no_meas_area"],
                                        cp_no_meas_area=self.param["cp_no_meas_area"],
                                        op_no_meas_area=self.param["op_no_meas_area"],
                                        up_no_meas_area=self.param["up_no_meas_area"],
                                        gw_no_meas_area=self.param["gw_no_meas_area"],
                                        gwl_prevt=lst_prevt["gwl"],
                                        delta_t=self.param["delta_t"])
            uz_sol = self.unsaturatedzone.sol(
                i_up_uz=up_sol["i_up_uz"],
                meas_uz=meas_sol["q_meas_uz"],
                tot_meas_area=self.param["tot_meas_area"],
                e_ref=ref_grass,
                gwl_prevt=lst_prevt["gwl"],
                delta_t=self.param["delta_t"],
            )
            gw_sol = self.groundwater.sol(
                p_uz_gw=uz_sol["p_uz_gw"],
                uz_no_meas_area=self.param["uz_no_meas_area"],
                p_op_gw=op_sol["p_op_gw"],
                op_no_meas_area=self.param["op_no_meas_area"],
                tot_meas_area=self.param["tot_meas_area"],
                meas_gw=meas_sol["q_meas_gw"],
                owl_prevt=lst_prevt["owl"],
                delta_t=self.param["delta_t"],
            )
            ss_sol = self.sewersystem.sol(
                pr_no_meas_area=self.param["pr_no_meas_area"],
                cp_no_meas_area=self.param["cp_no_meas_area"],
                op_no_meas_area=self.param["op_no_meas_area"],
                r_pr_swds=pr_sol["r_pr_swds"],
                r_cp_swds=cp_sol["r_cp_swds"],
                r_op_swds=op_sol["r_op_swds"],
                r_pr_mss=pr_sol["r_pr_mss"],
                r_cp_mss=cp_sol["r_cp_mss"],
                r_op_mss=op_sol["r_op_mss"],
                meas_swds=meas_sol["q_meas_swds"],
                meas_mss=meas_sol["q_meas_mss"],
                ow_no_meas_area=self.param["ow_no_meas_area"],
                tot_meas_area=self.param["tot_meas_area"],
            )
            ow_sol = self.openwater.sol(
                p_atm=p_atm,
                e_pot_ow=e_pot_ow,
                r_up_ow=up_sol["r_up_ow"],
                d_gw_ow=gw_sol["d_gw_ow"],
                q_swds_ow=ss_sol["q_swds_ow"],
                q_mss_ow=ss_sol["q_mss_ow"],
                so_swds_ow=ss_sol["so_swds_ow"],
                so_mss_ow=ss_sol["so_mss_ow"],
                meas_ow=meas_sol["q_meas_ow"],
                up_no_meas_area=self.param["up_no_meas_area"],
                gw_no_meas_area=self.param["gw_no_meas_area"],
                swds_no_meas_area=self.param["swds_no_meas_area"],
                mss_no_meas_area=self.param["mss_no_meas_area"],
                tot_meas_area=self.param["tot_meas_area"],
                tot_area=self.param["tot_area"],
                delta_t=self.param["delta_t"],
            )
            merged_dict = OrderedDict(dict(**pr_sol, **cp_sol, **op_sol, **up_sol, **uz_sol, **gw_sol, **ss_sol,
                                           **ow_sol, **meas_sol))
        except IndexError:
            raise StopIteration
        return merged_dict


def read_inputdata(dyn_inp):
    """
    reads input data (time series of precipitation and evaporation) from dynamic input file.

    Args:
        dyn_inp (string): the filename of the input time series of precipitation and evaporation

    Returns:
        (dataframe): A dataframe of the time series of precipitation and evaporation
    """
    path = Path.cwd() / ".." / "input"

    # Parsing 30-year date data as datetime will takes additional 25 seconds.
    # Besides, the actual user-defined datetime format can be problematic.
    # Therefore, date is not parsed as datetime here.
    # rv = pd.read_csv(str(path) + "\\" + dyn_inp, parse_dates=["date"], dayfirst=True)
    rv = pd.read_csv(str(path) + "\\" + dyn_inp)

    # check if there is missing value in the input time series.
    num_nan = rv.isnull().sum().sum()
    if num_nan != 0:
        raise SystemExit(f"There are {num_nan} missing values in time series. Please recheck input time series file.")
    return rv


def timer(func):
    """
    a decorator that timings the function runtime.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        print(f"Elapsed: {after - start:.2f}s")
        return rv
    return wrapper

def read_parameters(stat1_inp, stat2_inp):
    """
    reads parameters for model initialization by calling "read_parameter_base" to read parameters from neighbourhood
    configuration file, calling "read_parameter_measure" to read parameters from measure configuration file, and
    computing area of xx without measure with given parameters.

    Args:
        stat1_inp (string): filename of neighbourhood configuration file
        stat2_inp (string): filename of measure configuration file

    Returns:
        (dictionary): A dictionary of all necessary parameters to initialize a model
    """
    parameter_base = read_parameter_base(stat1_inp)
    parameter_measure = read_parameter_measure(stat2_inp)
    d = dict(pr_no_meas_area=parameter_base["tot_pr_area"] - parameter_measure["pr_meas_area"],
             cp_no_meas_area=parameter_base["tot_cp_area"] - parameter_measure["cp_meas_area"],
             op_no_meas_area=parameter_base["tot_op_area"] - parameter_measure["op_meas_area"],
             up_no_meas_area=parameter_base["tot_up_area"] - parameter_measure["up_meas_area"],
             uz_no_meas_area=parameter_base["tot_uz_area"] - parameter_measure["uz_meas_area"],
             gw_no_meas_area=parameter_base["tot_gw_area"] - parameter_measure["gw_meas_area"],
             swds_no_meas_area=parameter_base["tot_swds_area"] - parameter_measure["swds_meas_area"],
             mss_no_meas_area=parameter_base["tot_mss_area"] - parameter_measure["mss_meas_area"],
             ow_no_meas_area=parameter_base["tot_ow_area"] - parameter_measure["ow_meas_area"],
             )
    rv = {**parameter_base, **parameter_measure, **d}
    # print(rv)
    return rv


def check_parameters(dict_param):
    """
    especially used in batch_run_measure() when the simulation switches from "with measure" cases to "without measure",
    i.e. the baseline case, in order to make sure all area-related parameters are correctly modified accordingly.

    Args:
        dict_param (dictionary): a dictionary of parameters to initialize a model, which needs to be checked.

    Returns:
        (dictionary): A dictionary of all necessary parameters to initialize a model
    """
    if dict_param["measure_applied"]:
        return dict_param
    else:
        # update area-related parameters:
        # measure inflow area
        dict_param["pr_meas_inflow_area"] = dict_param["cp_meas_inflow_area"] = dict_param["op_meas_inflow_area"] = \
            dict_param["up_meas_inflow_area"] = dict_param["ow_meas_inflow_area"] = 0.0
        dict_param["tot_meas_inflow_area"] = 0.0
        # area of xx with measure
        dict_param["pr_meas_area"] = dict_param["cp_meas_area"] = dict_param["op_meas_area"] = \
            dict_param["up_meas_area"] = dict_param["uz_meas_area"] = dict_param["gw_meas_area"] = \
            dict_param["swds_meas_area"] = dict_param["mss_meas_area"] = dict_param["ow_meas_area"] = 0.0
        # area of interception layer, top storage layer, and bottom storage layer of measure
        dict_param["tot_meas_area"] = dict_param["top_meas_area"] = dict_param["btm_meas_area"] = 0.0
        # print(dict_param)

        # dictionary of area of xx without measure
        d = dict(pr_no_meas_area=dict_param["tot_pr_area"] - dict_param["pr_meas_area"],
                 cp_no_meas_area=dict_param["tot_cp_area"] - dict_param["cp_meas_area"],
                 op_no_meas_area=dict_param["tot_op_area"] - dict_param["op_meas_area"],
                 up_no_meas_area=dict_param["tot_up_area"] - dict_param["up_meas_area"],
                 uz_no_meas_area=dict_param["tot_uz_area"] - dict_param["uz_meas_area"],
                 gw_no_meas_area=dict_param["tot_gw_area"] - dict_param["gw_meas_area"],
                 swds_no_meas_area=dict_param["tot_swds_area"] - dict_param["swds_meas_area"],
                 mss_no_meas_area=dict_param["tot_mss_area"] - dict_param["mss_meas_area"],
                 ow_no_meas_area=dict_param["tot_ow_area"] - dict_param["ow_meas_area"],
                 )
        # updates dict_param with values in d
        rv = {**dict_param, **d}
        return rv


#@timer
def running(input_data, dict_param):
    """
    a basic running unit, which takes the forcing from input_data and the parameters from a dictionary of parameters to
    run the simulation once and returns all the results in a dataframe. After calculation, the water balance for the
    entire model, measure itself, and measure inflow area is checked and the corresponding statistics is returned.

    Args:
        input_data (dataframe): a fixed-format dataframe of time series of forcing (precipitation and evaporation)
        dict_param (dictionary): a dictionary of all necessary parameters to initialize a model

    Returns:
        (dataframe): A dataframe of computed results for all time steps
    """
    # global unit_list
    date = input_data["date"]
    P_atm = input_data["P_atm"]
    Ref_grass = input_data["Ref.grass"]
    E_pot_OW = input_data["E_pot_OW"]
    iters = np.shape(date)[0]
    # # print area-related parameters in the dictionary for checking. Uncomment it when necessary.
    # print("tot_area", dict_param["tot_area"])
    # print("tot_pr_area", dict_param["tot_pr_area"], "pr_no_meas_area", dict_param["pr_no_meas_area"], "pr_meas_area", dict_param["pr_meas_area"], "pr_meas_inflow_area", dict_param["pr_meas_inflow_area"])
    # print("tot_cp_area", dict_param["tot_cp_area"], "cp_no_meas_area", dict_param["cp_no_meas_area"], "cp_meas_area", dict_param["cp_meas_area"], "cp_meas_inflow_area", dict_param["cp_meas_inflow_area"])
    # print("tot_op_area", dict_param["tot_op_area"], "op_no_meas_area", dict_param["op_no_meas_area"], "op_meas_area", dict_param["op_meas_area"], "op_meas_inflow_area", dict_param["op_meas_inflow_area"])
    # print("tot_up_area", dict_param["tot_up_area"], "up_no_meas_area", dict_param["up_no_meas_area"], "up_meas_area", dict_param["up_meas_area"], "up_meas_inflow_area", dict_param["up_meas_inflow_area"])
    # print("tot_uz_area", dict_param["tot_uz_area"], "uz_no_meas_area", dict_param["uz_no_meas_area"], "uz_meas_area", dict_param["uz_meas_area"],)
    # print("tot_gw_area", dict_param["tot_gw_area"], "gw_no_meas_area", dict_param["gw_no_meas_area"], "gw_meas_area", dict_param["gw_meas_area"],)
    # print("tot_swds_area", dict_param["tot_swds_area"], "swds_no_meas_area", dict_param["swds_no_meas_area"], "swds_meas_area", dict_param["swds_meas_area"],)
    # print("tot_mss_area", dict_param["tot_mss_area"], "mss_no_meas_area", dict_param["mss_no_meas_area"], "mss_meas_area", dict_param["mss_meas_area"],)
    # print("tot_ow_area", dict_param["tot_ow_area"], "ow_no_meas_area", dict_param["ow_no_meas_area"], "ow_meas_area", dict_param["ow_meas_area"], "ow_meas_inflow_area", dict_param["ow_meas_inflow_area"])
    # print("tot_meas_area", dict_param["tot_meas_area"], "top_meas_area", dict_param["top_meas_area"], "btm_meas_area", dict_param["btm_meas_area"])

    # create an instance of the UrbanwbModel class
    k = UrbanwbModel(dict_param)
    # first row of dataframe that stores initial values
    lst = [
        {
            "int_pr": np.nan,
            "e_atm_pr": np.nan,
            "intstor_pr": dict_param["intstor_pr_t0"],
            "r_pr_meas": np.nan,
            "r_pr_swds": np.nan,
            "r_pr_mss": np.nan,
            "r_pr_up": np.nan,
            "int_cp": np.nan,
            "e_atm_cp": np.nan,
            "intstor_cp": dict_param["intstor_cp_t0"],
            "r_cp_meas": np.nan,
            "r_cp_swds": np.nan,
            "r_cp_mss": np.nan,
            "r_cp_up": np.nan,
            "int_op": np.nan,
            "e_atm_op": np.nan,
            "intstor_op": dict_param["intstor_op_t0"],
            "p_op_gw": np.nan,
            "r_op_meas": np.nan,
            "r_op_swds": np.nan,
            "r_op_mss": np.nan,
            "r_op_up": np.nan,
            "sum_r_up": np.nan,
            "init_intstor_up": np.nan,
            "actl_infilcap_up": np.nan,
            "timefac_up": np.nan,
            "e_atm_up": np.nan,
            "i_up_uz": np.nan,
            "fin_intstor_up": dict_param["fin_intstor_up_t0"],
            "r_up_meas": np.nan,
            "r_up_ow": np.nan,
            "sum_i_uz": np.nan,
            "r_meas_uz": np.nan,
            "theta_h3_uz": np.nan,
            "t_alpha_uz": np.nan,
            "t_atm_uz": np.nan,
            "gwl_up": np.nan,
            "gwl_low": np.nan,
            "theta_eq_uz": np.nan,
            "capris_max_uz": np.nan,
            "p_uz_gw": np.nan,
            "theta_uz": soil_selector(dict_param["soiltype"], dict_param["croptype"])[
                gwlcalc(dict_param["gwl_t0"])[2]
            ]["moist_cont_eq_rz[mm]"],
            "sum_p_gw": np.nan,
            "r_meas_gw": np.nan,
            "sc_gw": soil_selector(dict_param["soiltype"], dict_param["croptype"])[
                gwlcalc(dict_param["gwl_t0"])[2]
            ]["stor_coef"],
            "h_gw": np.nan,
            "s_gw_out": np.nan,
            "d_gw_ow": np.nan,
            "gwl": dict_param["gwl_t0"],
            "gwl_sl": 0,
            "sum_r_swds": np.nan,
            "r_meas_swds": np.nan,
            "sum_r_mss": np.nan,
            "r_meas_mss": np.nan,
            "q_swds_ow": np.nan,
            "q_mss_out": np.nan,
            "q_mss_ow": np.nan,
            "so_swds_ow": dict_param["so_swds_t0"],
            "so_mss_ow": dict_param["so_mss_t0"],
            "stor_swds": dict_param["stor_swds_t0"],
            "stor_mss": dict_param["stor_mss_t0"],
            "prec_ow": np.nan,
            "e_atm_ow": np.nan,
            "sum_r_ow": np.nan,
            "sum_d_ow": np.nan,
            "sum_q_ow": np.nan,
            "sum_so_ow": np.nan,
            "r_meas_ow": np.nan,
            "q_ow_out": np.nan,
            "owl": dict_param["ow_level"],
            "prec_meas": np.nan,
            "sum_r_meas": np.nan,
            "int_meas": np.nan,
            "e_atm_meas": np.nan,
            "interc_down_meas": np.nan,
            "surf_runoff_meas": np.nan,
            "intstor_meas": dict_param["intstor_meas_t0"],
            "ini_stor_top_meas": np.nan,
            "t_atm_top_meas": np.nan,
            "perc_top_meas": np.nan,
            "fin_stor_top_meas": dict_param["stor_top_meas_t0"],
            "ini_stor_btm_meas": np.nan,
            "t_atm_btm_meas": np.nan,
            "p_gw_btm_meas": np.nan,
            "runoff_btm_meas": np.nan,
            "fin_stor_btm_meas": dict_param["stor_btm_meas_t0"],
            "overflow_btm_meas": np.nan,
            "q_meas_ow": np.nan,
            "q_meas_uz": np.nan,
            "q_meas_gw": np.nan,
            "q_meas_swds": np.nan,
            "q_meas_mss": np.nan,
            "q_meas_out": np.nan,
            # "uncontrolled_runoff": np.nan,
            # "controlled_runoff": np.nan,
            # "total_runoff": np.nan
        }
    ]
    for t in range(1, iters):  # time series first line is not relevant, computation starts from the second line.
        lst.append(
            k.__next__(
            P_atm[t],
            E_pot_OW[t],
            Ref_grass[t],
            lst[t - 1],
                        )
                    )
    df = pd.DataFrame(lst)
    df.insert(0, "Date", date)
    df.insert(1, "P_atm", P_atm)
    df.insert(2, "E_pot_OW", E_pot_OW)
    df.insert(3, "Ref.grass", Ref_grass)
    wbc_results = water_balance_checker_no_print(df, dict_param, iters)
    return df, wbc_results


def save_to_csv(dyn_inp, stat1_inp, stat2_inp, output_filename, *args, save_all=True):
    """
    runs the simulation with three files (csv file of time series, configuration files of neighbourhood(base) and
    measure) and saves results in a csv file with the specified output filename under the 'pysol' folder.

    Args:
        dyn_inp (string): the filename of the dynamic input data of precipitation and evaporation
        stat1_inp (string): the filename of the static form of general parameters
        stat2_inp (string): the filename of the static form of measure parameters
        output_filename (string): the filename of the output file of solutions
        *args (strings): specified selected results to be saved
        save_all (bool): save all results when True, save specified selected results when False

    Returns:
        A csv file of all (or part of) computed results
    """
    loggingfilename = ''.join(list(output_filename)[:-4]) + ".log"
    logger = setuplog(loggingfilename, "STC_logger", thelevel=logging.INFO)

    input_data = read_inputdata(dyn_inp)
    dict_param = read_parameters(stat1_inp, stat2_inp)
    rv = running(input_data, dict_param)
    df = rv[0]
    wbc_statistics = rv[1]
    # logging the water balance for entire model
    logger.info(f"Entire model: {wbc_statistics[0]}")
    # logging the water balance for measure itself
    logger.info(f"Measure itself: {wbc_statistics[1]}")

    # if measure is applied, logging the water balance for measure inflow area
    if dict_param["tot_meas_area"] != 0:
        logger.info(f"Measure inflow area: {wbc_statistics[2]}")

    # if warning messages is not empty, logging the warning message
    if len(wbc_statistics[3]) != 0:
        logger.warning(wbc_statistics[3])

    outdir = Path("pysol")
    outdir.mkdir(parents=True, exist_ok=True)

    # logging the info regarding saving
    if save_all:
        msg = f"Saving all results to {output_filename}..."
    else:
        msg = f"Saving results {args} to {output_filename}..."
    logger.info(msg)
    print(msg)

    if save_all:
        df.to_csv(outdir / output_filename, index=True)
    else:
        header = ["Date", "P_atm", "E_pot_OW", "Ref.grass"]
        header.extend([arg for arg in args])
        df.to_csv(outdir / output_filename, index=True, columns=header)

#
# def batch_run_single(dyn_inp, stat1_inp, stat2_inp, dyn_out, varkey, *vararr):
#     """
#     this batch_run function is to batch-run specified parameter with a set of parameters and save all results in csv
#     for every case.
#
#     Args:
#         dyn_inp (string): the filename of the inputdata of precipitation and evaporation
#         stat1_inp (string): the filename of the static form of general parameters
#         stat2_inp (string): the filename of the static form of measure parameters
#         dyn_out (string): the general filename of the output file of solutions
#         varkey (string): the parameter that needs to be updated in the batch run.
#         vararr (float): values to update varkey.
#     """
#     import os
#     param = {**read_parameter_base(stat1_inp), **read_parameter_measure(stat2_inp)}
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#     for varval in vararr:
#         param[str(varkey)] = varval
#         df = run(param, dyn_inp)
#         new_dyn_out = f"{varkey}={varval}_" + dyn_out
#         fullname = os.path.join(outdir, new_dyn_out)
#         df.to_csv(fullname, index=True)

#
# def batch_run_save_to_csv1(dyn_inp, stat1_inp, stat2_inp, output_filename, param_to_change, *args):
#     """
#     this batch run function runs the model with a set of specified parameters and save all results in seperated csv
#     """
#
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     input_data = read_inputdata(dyn_inp)
#     dict_param = read_parameters(stat1_inp, stat2_inp)
#     for arg in args:
#         dict_param[param_to_change] = arg
#         df = running(input_data, dict_param)[0]
#         output = str(arg) + output_filename
#         df.to_csv(outdir / output, index=True)


# def batch_run_save_to_csv2(dyn_inp, stat1_inp, stat2_inp, output_filename, param_to_change, value_list, corresponding_varkey, value_list2):
#     """
#     this batch run function runs the model with a set of specified parameters and save all results in seperated csv"""
#
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     input_data = read_inputdata(dyn_inp)
#     dict_param = read_parameters(stat1_inp, stat2_inp)
#     for x, y in zip(value_list, value_list2):
#         dict_param[param_to_change] = x
#         dict_param[corresponding_varkey] = y
#         df = running(input_data, dict_param)[0]
#         output = str(x) + output_filename
#         df.to_csv(outdir / output, index=True)


# def batch_run2(dyn_inp, stat1_inp, stat2_inp, dyn_out, varkey, *vararr, *col, saveall=True):
#     """
#     this batch_run function is to batch-run specified parameter with a set of parameters and save all results in csv
#     for every case.
#
#     Args:
#         dyn_inp (string): the filename of the inputdata of precipitation and evaporation
#         stat1_inp (string): the filename of the static form of general parameters
#         stat2_inp (string): the filename of the static form of measure parameters
#         dyn_out (string): the general filename of the output file of solutions
#         varkey (string): the parameter that needs to be updated in the batch run.
#         vararr (float): values to update varkey.
#     """
#     import os
#     param = {**read_parameter_base(stat1_inp), **read_parameter_measure(stat2_inp)}
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#     for varval in vararr:
#         param[str(varkey)] = varval
#         df = run(param, dyn_inp)
#         new_dyn_out = f"{varkey}={varval}_" + dyn_out
#         fullname = os.path.join(outdir, new_dyn_out)
#         df.to_csv(fullname, index=True)


def batch_run_measure(dyn_inp, stat1_inp, stat2_inp, dyn_out, varkey, vararrlist1, correspvarkey=None, vararrlist2=None,
                      baseline_variable="r_op_swds", variable_to_save="q_meas_swds"):
    """
    for one type of measure, run a batch of simulations with different values for one (or two) parameter(s)

    Args:
    dyn_inp (string): the filename of the inputdata of precipitation and evaporation
    stat1_inp (string): the filename of the static form of general parameters
    stat2_inp (string): the filename of the static form of measure parameters
    dyn_out (string): the filename of the output file of solutions
    varkey (float): the key parameter to be updated
    vararr (float): values to update varkey

    Usage:
    use in the cmd: python -m urbanwb.main batch_run_measure timeseries.csv stat1.ini stat2.ini results.csv storcap_btm_meas [20,30,40]
    """
    loggingfilename = ''.join(list(dyn_out)[:-4]) + ".log"
    logger = setuplog(loggingfilename, "BRM_logger", thelevel=logging.INFO)
    inputdata = read_inputdata(dyn_inp)
    dict_param = read_parameters(stat1_inp, stat2_inp)

    outdir = Path("pysol")
    outdir.mkdir(parents=True, exist_ok=True)

    # can delete this fraction if necessary.
    date = inputdata["date"]
    iters = np.shape(date)[0]
    dt = dict_param["delta_t"]
    num_year = round((dt * iters) / 365)
    print(f"Total year of the input time series is {num_year} year")
    nameofmeasure = dict_param["title"]
    msg_nameofmeasure = f"Current running {nameofmeasure}"
    logger.info(msg_nameofmeasure)
    print(msg_nameofmeasure)
    print("\n")
    database = []
    if correspvarkey is not None:
        for a, b in zip(vararrlist1, vararrlist2):
            dict_param[varkey] = a
            dict_param[correspvarkey] = b
            measure_area_info = dict_param["tot_meas_area"]
            measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
            msg = f"Case with measure: {varkey}={a}, {correspvarkey}={b}, measure area={measure_area_info}, " \
                f"inflow area={measure_inflow_area_info}"
            print(msg)
            rv = running(inputdata, dict_param)
            database.append(pd.DataFrame(rv[0])[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"])
            logger.info(msg)
            wbc_statistics = rv[1]
            logger.info(f"Entire model: {wbc_statistics[0]}")
            logger.info(f"Measure itself: {wbc_statistics[1]}")
            logger.info(f"Measure inflow area: {wbc_statistics[2]}")
            print("------" * 20)
            print("\n"*2)
            sleep(0.5)
    else:
        for a in vararrlist1:
            dict_param[varkey] = a
            measure_area_info = dict_param["tot_meas_area"]
            measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
            msg = f"Case with measure: {varkey}={a},measure area={measure_area_info}, inflow area={measure_inflow_area_info}"
            print(msg)
            rv = running(inputdata, dict_param)
            database.append(pd.DataFrame(rv[0])[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"])
            logger.info(msg)
            wbc_statistics = rv[1]
            logger.info(f"Entire model: {wbc_statistics[0]}")
            logger.info(f"Measure itself: {wbc_statistics[1]}")
            logger.info(f"Measure inflow area: {wbc_statistics[2]}")
            print("------" * 20)
            print("\n" * 2)
            sleep(0.5)

    df = pd.DataFrame(database, index=[v for v in vararrlist1])
    df = df.T
    df.insert(0, "Date", date)
    df.insert(1, "P_atm", inputdata["P_atm"])

    dict_param["measure_applied"] = False
    # print(dict_param)
    msg = "Case without measure: Baseline"
    print(msg)
    rv = running(inputdata, check_parameters(dict_param))
    baseline_runoff = pd.DataFrame(rv[0])[baseline_variable]
    logger.info(msg)
    wbc_statistics = rv[1]
    logger.info(f"Entire model: {wbc_statistics[0]}")
    logger.info(f"Measure itself: {wbc_statistics[1]}")
    # logger.info(f"Measure' impact over measure inflow area: {wbc_statistics[2]}")
    print("------" * 20)
    sleep(0.5)
    df.insert(2, "Baseline", baseline_runoff)
    outdir = Path("pysol")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / dyn_out, index=True)


def batch_run_sdf(dyn_inp, stat1_inp, stat2_inp, dyn_out, q_list, baseline_q=None, arithmetic_progression=False):
    """
    this batch_run function is mainly designed for getting the database for sdf_curve.

    Args:
        dyn_inp (string): the filename of the inputdata of precipitation and evaporation
        stat1_inp (string): the filename of the static form of general parameters
        stat2_inp (string): the filename of the static form of measure parameters
        dyn_out (string): the filename of the output file of solutions
        q_list (float): a list of values to update "q_ow_out_cap"
        baseline_q (float): default baseline q is mean daily rainfall if baseline_q not explicitly defined
        arithmetic_progression (bool): default is False meaning q_list is a list of random number; when explicitly
        defined True, q_list is (min,max,steps)

    """
    # determine logfile name based on outputfile name
    #loggingfilename = ''.join(list(dyn_out)[:-4]) + ".log"
    #logger = setuplog(loggingfilename, "BRSDF_logger", thelevel=logging.INFO)
    input_data = read_inputdata(dyn_inp)
    dict_param = read_parameters(stat1_inp, stat2_inp)

    outdir = Path("pysol")
    outdir.mkdir(parents=True, exist_ok=True)

    rank_database = []
    iters = len(input_data["date"])
    dt = dict_param["delta_t"]
    mean_daily_rainfall = np.mean(input_data["P_atm"])/dt
    num_year = round((dt * iters) / 365)
#    print(f"The length of input time series is around {num_year} year")
#    print(f"Mean daily rainfall is {mean_daily_rainfall:.2f} mm/d")
#    print("First, do baseline run:")

    if baseline_q is None:
        dict_param["q_ow_out_cap"] = mean_daily_rainfall
#        msg0 = f"Baseline pumping capacity is by default set as mean daily rainfall {mean_daily_rainfall:.2f} mm/d to make fixed marks"
#        logger.info(msg0)
#        print(msg0)
    else:
#        msg0 = f"Baseline pumping capacity is {baseline_q} mm/d to make fixed marks"
        dict_param["q_ow_out_cap"] = baseline_q
#        logger.info(msg0)
#        print(msg0)

    # perform baseline run
    owl_data = np.append(running(input_data, dict_param)[0]["owl"], 0)  # extra 0 at the end
    owl_baseline = np.ones(len(owl_data)) * dict_param["ow_level"] - owl_data
    segment_marks = get_segment_index(owl_baseline)
    k_base = SDF_curve2(segment_marks, owl_data, ow_level=dict_param["ow_level"])
    rank_database.append(k_base.ranking)
    # print(segment_marks)
#    print("-----"*50)
    if not arithmetic_progression:  # if it is random number to type in.
#        print(f"q value to batch run: {q_list}")
        for q in q_list:
            dict_param["q_ow_out_cap"] = q
#            msg1 = f"Running: pumping capacity from open water to outside is {q} mm/d over entire area"
#            print(msg1)
#            logger.info(msg1)
            rv = running(input_data, dict_param)
            wbc_statistics = rv[1]
            # logging the water balance for entire model
#            logger.info(f"Entire model: {wbc_statistics[0]}")
#            # logging the water balance for measure itself
#            logger.info(f"Measure itself: {wbc_statistics[1]}")
            # if measure is applied, logging the water balance for measure inflow area
#            if dict_param["tot_meas_area"] != 0:
#                logger.info(f"Measure inflow area: {wbc_statistics[2]}")
            # if warning messages is not empty, logging the warning message
#            if len(wbc_statistics[3]) != 0:
#                logger.warning(wbc_statistics[3])
            owl_data = pd.DataFrame(rv[0])["owl"]
            k = SDF_curve2(segment_marks, owl_data, ow_level=dict_param["ow_level"])
            rank_database.append(k.ranking)
#            msg2 = f"Maximum storage height above target water level over open water for Q = {q} mm/d is {k.ranking[0]:.4f} m"
#            print(msg2)
#            logger.info(msg2)
#            print("-----"*50)
        if baseline_q is None:
            name_of_index = [f"{mean_daily_rainfall:.2f}"] + [f"{v}" for v in q_list]
        else:
            name_of_index = [f"{baseline_q:.2f}"] + [f"{v}" for v in q_list]
        df = pd.DataFrame(rank_database, index=name_of_index)
        #outdir = Path("pysol")
        #outdir.mkdir(parents=True, exist_ok=True)
        #df.T.to_csv(outdir / dyn_out, index=True)

    else:  # if we type in an arithmetic progression
        if len(q_list) != 3:
            raise SystemExit("Please type in min, max, steps.")
        array_q = np.arange(q_list[0], q_list[1], (q_list[1] - q_list[0]) / q_list[2])
        array_q = np.append(array_q, q_list[1])
#        print(f"q value to batch run are {array_q}")
        for q in array_q:
            dict_param["q_ow_out_cap"] = q
#            msg1 = f"Running: pumping capacity from open water to outside is {q} mm/d over entire area"
#            print(msg1)
#            logger.info(msg1)
            rv = running(input_data, dict_param)
            wbc_statistics = rv[1]
            # logging the water balance for entire model
#            logger.info(f"Entire model: {wbc_statistics[0]}")
#            # logging the water balance for measure itself
#            logger.info(f"Measure itself: {wbc_statistics[1]}")
            # if measure is applied, logging the water balance for measure inflow area
#            if dict_param["tot_meas_area"] != 0:
#                logger.info(f"Measure inflow area: {wbc_statistics[2]}")
            # if warning messages is not empty, logging the warning message
#            if len(wbc_statistics[3]) != 0:
#                logger.warning(wbc_statistics[3])
            owl_data = pd.DataFrame(rv[0])["owl"]
            k = SDF_curve2(segment_marks, owl_data, ow_level=dict_param["ow_level"])
            rank_database.append(k.ranking)
#            msg2 = f"Maximum storage height above target water level over open water for Q = {q} mm/d is {k.ranking[0]:.4f} m"
#            print(msg2)
#            logger.info(msg2)
#            print("-----" * 40)

        if baseline_q is None:
            name_of_index = [f"{mean_daily_rainfall:.2f}"] + [f"{v}" for v in array_q]
        else:
            name_of_index = [f"{baseline_q:.2f}"] + [f"{v}" for v in array_q]

        df = pd.DataFrame(rank_database, index=name_of_index)
    return df
#        outdir = Path("pysol")
#        outdir.mkdir(parents=True, exist_ok=True)
#        df.T.to_csv(outdir / dyn_out, index=True)


# def batch_run_measure_mia(dyn_inp, stat1_inp, stat2_inp, dyn_out, varkey, vararrlist1, correspvarkey=None, vararrlist2=None,
#                       baseline_variable="r_op_swds", variable_to_save="controlled_runoff"):
#     """
#     for one type of measure, run a batch of simulations with different values for one (or two) parameter(s)
#
#     Args:
#     dyn_inp (string): the filename of the inputdata of precipitation and evaporation
#     stat1_inp (string): the filename of the static form of general parameters
#     stat2_inp (string): the filename of the static form of measure parameters
#     dyn_out (string): the filename of the output file of solutions
#     varkey (float): the key parameter to be updated
#     vararr (float): values to update varkey
#
#     Usage:
#     use in the cmd: python -m urbanwb.main_with_measure batch_run_measure_ctrl timeseries.csv stat1.ini stat2.ini results.csv storcap_btm_meas [20,30,40]
#     """
#     loggingfilename = ''.join(list(dyn_out)[:-4]) + ".log"
#     logger = setuplog(loggingfilename, "BRM_logger", thelevel=logging.INFO)
#     inputdata = read_inputdata(dyn_inp)
#     dict_param = read_parameters(stat1_inp, stat2_inp)
#
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     # can delete this fraction if necessary.
#     date = inputdata["date"]
#     iters = np.shape(date)[0]
#     dt = dict_param["delta_t"]
#     num_year = round((dt * iters) / 365)
#     print(f"Total year of the input time series is {num_year} year")
#     nameofmeasure = dict_param["title"]
#     msg_nameofmeasure = f"Current running {nameofmeasure}"
#     logger.info(msg_nameofmeasure)
#     print(msg_nameofmeasure)
#     print("\n")
#     database = []
#     if correspvarkey is not None:
#         for a, b in zip(vararrlist1, vararrlist2):
#             dict_param[varkey] = a
#             dict_param[correspvarkey] = b
#             measure_area_info = dict_param["tot_meas_area"]
#             measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
#             msg = f"Case with measure: {varkey}={a}, {correspvarkey}={b}, measure area={measure_area_info}, " \
#                 f"inflow area={measure_inflow_area_info}"
#             print(msg)
#             rv = running(inputdata, dict_param)
#             database.append(pd.DataFrame(rv[0])[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"])
#             logger.info(msg)
#             wbc_statistics = rv[1]
#             logger.info(f"Entire model: {wbc_statistics[0]}")
#             logger.info(f"Measure itself: {wbc_statistics[1]}")
#             logger.info(f"Measure inflow area: {wbc_statistics[2]}")
#             print("------" * 20)
#             print("\n"*2)
#             sleep(0.5)
#     else:
#         for a in vararrlist1:
#             dict_param[varkey] = a
#             measure_area_info = dict_param["tot_meas_area"]
#             measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
#             msg = f"Case with measure: {varkey}={a},measure area={measure_area_info}, inflow area={measure_inflow_area_info}"
#             print(msg)
#             rv = running(inputdata, dict_param)
#             database.append(pd.DataFrame(rv[0])[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"])
#             logger.info(msg)
#             wbc_statistics = rv[1]
#             logger.info(f"Entire model: {wbc_statistics[0]}")
#             logger.info(f"Measure itself: {wbc_statistics[1]}")
#             logger.info(f"Measure inflow area: {wbc_statistics[2]}")
#             print("------" * 20)
#             print("\n" * 2)
#             sleep(0.5)
#
#     df = pd.DataFrame(database, index=[v for v in vararrlist1])
#     df = df.T
#     df.insert(0, "Date", date)
#     df.insert(1, "P_atm", inputdata["P_atm"])
#
#     dict_param["measure_applied"] = False
#     # print(dict_param)
#     msg = "Case without measure: Baseline"
#     print(msg)
#     rv = running(inputdata, check_parameters(dict_param))
#     baseline_runoff = pd.DataFrame(rv[0])[baseline_variable]
#     logger.info(msg)
#     wbc_statistics = rv[1]
#     logger.info(f"Entire model: {wbc_statistics[0]}")
#     logger.info(f"Measure itself: {wbc_statistics[1]}")
#     # logger.info(f"Measure' impact over measure inflow area: {wbc_statistics[2]}")
#     print("------" * 20)
#     sleep(0.5)
#     df.insert(2, "Baseline", baseline_runoff)
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#     df.to_csv(outdir / dyn_out, index=True)
#
#
# def batch_run_measure_tot_area(dyn_inp, stat1_inp, stat2_inp, dyn_out, varkey, vararrlist1, correspvarkey=None, vararrlist2=None, variable_to_save="r_ow_entire3"):
#     """
#     for one type of measure, run a batch of simulations with different values for one (or two) parameter(s)
#
#     Args:
#     dyn_inp (string): the filename of the inputdata of precipitation and evaporation
#     stat1_inp (string): the filename of the static form of general parameters
#     stat2_inp (string): the filename of the static form of measure parameters
#     dyn_out (string): the filename of the output file of solutions
#     varkey (float): the key parameter to be updated
#     vararr (float): values to update varkey
#
#     Usage:
#     use in the cmd: python -m urbanwb.main_with_measure batch_run_measure_ctrl timeseries.csv stat1.ini stat2.ini results.csv storcap_btm_meas [20,30,40]
#     """
#     loggingfilename = ''.join(list(dyn_out)[:-4]) + ".log"
#     logger = setuplog(loggingfilename, "BRM_logger", thelevel=logging.INFO)
#     inputdata = read_inputdata(dyn_inp)
#     dict_param = read_parameters(stat1_inp, stat2_inp)
#
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     # can delete this fraction if necessary.
#     date = inputdata["date"]
#     iters = np.shape(date)[0]
#     dt = dict_param["delta_t"]
#     num_year = round((dt * iters) / 365)
#     print(f"Total year of the input time series is {num_year} year")
#     nameofmeasure = dict_param["title"]
#     msg_nameofmeasure = f"Current running {nameofmeasure}"
#     logger.info(msg_nameofmeasure)
#     print(msg_nameofmeasure)
#     print("\n")
#     database = []
#     if correspvarkey is not None:
#         for a, b in zip(vararrlist1, vararrlist2):
#             dict_param[varkey] = a
#             dict_param[correspvarkey] = b
#             measure_area_info = dict_param["tot_meas_area"]
#             measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
#             msg = f"Case with measure: {varkey}={a}, {correspvarkey}={b}, measure area={measure_area_info}, " \
#                 f"inflow area={measure_inflow_area_info}"
#             print(msg)
#             rv = running(inputdata, dict_param)
#             database.append(pd.DataFrame(rv[0])[variable_to_save])
#             logger.info(msg)
#             wbc_statistics = rv[1]
#             logger.info(f"Entire model: {wbc_statistics[0]}")
#             logger.info(f"Measure itself: {wbc_statistics[1]}")
#             logger.info(f"Measure inflow area: {wbc_statistics[2]}")
#             print("------" * 20)
#             print("\n"*2)
#             sleep(0.5)
#     else:
#         for a in vararrlist1:
#             dict_param[varkey] = a
#             measure_area_info = dict_param["tot_meas_area"]
#             measure_inflow_area_info = dict_param["tot_meas_inflow_area"]
#             msg = f"Case with measure: {varkey}={a},measure area={measure_area_info}, inflow area={measure_inflow_area_info}"
#             print(msg)
#             rv = running(inputdata, dict_param)
#             database.append(pd.DataFrame(rv[0])[variable_to_save])
#             logger.info(msg)
#             wbc_statistics = rv[1]
#             logger.info(f"Entire model: {wbc_statistics[0]}")
#             logger.info(f"Measure itself: {wbc_statistics[1]}")
#             logger.info(f"Measure inflow area: {wbc_statistics[2]}")
#             print("------" * 20)
#             print("\n" * 2)
#             sleep(0.5)
#
#     df = pd.DataFrame(database, index=[v for v in vararrlist1])
#     df = df.T
#     df.insert(0, "Date", date)
#     df.insert(1, "P_atm", inputdata["P_atm"])
#     # df.insert(2, "evap", inputdata["E_pot_OW"])
#
#     dict_param["measure_applied"] = False
#     # print(dict_param)
#     msg = "Case without measure: Baseline"
#     print(msg)
#     rv = running(inputdata, check_parameters(dict_param))
#     baseline_runoff = pd.DataFrame(rv[0])[variable_to_save]
#     # gwl_ts = pd.DataFrame(rv[0])["gwl"]
#     # owl_ts = pd.DataFrame(rv[0])["owl"]
#     # q_ow_out_ts = pd.DataFrame(rv[0])["q_ow_out"]
#     logger.info(msg)
#     wbc_statistics = rv[1]
#     logger.info(f"Entire model: {wbc_statistics[0]}")
#     logger.info(f"Measure itself: {wbc_statistics[1]}")
#     # logger.info(f"Measure' impact over measure inflow area: {wbc_statistics[2]}")
#     print("------" * 20)
#     sleep(0.5)
#     df.insert(2, "Baseline", baseline_runoff)
#     # df.insert(4, "GWL", gwl_ts)
#     # df.insert(5, "owl", owl_ts)
#     # df.insert(6, "q_ow_out", q_ow_out_ts)
#     outdir = Path("pysol")
#     outdir.mkdir(parents=True, exist_ok=True)
#     df.to_csv(outdir / dyn_out, index=True)

# =============================================================================
# Added/edited functions
# =============================================================================

def water_balance_checker_no_print(df, dict_param, iters):
    """
    checks whether water balance is closed both over entire area and over measure itself
    Edited: does not print the water balance statistics when looping over the measures
    """

    # check water balance over entire area:
#    print("Water balance statistics: ")
    warning_msgs = []

    # precipitation over entire area
    sum_prec = sum(df["P_atm"].iloc[1:])

    # evaporation of xx over entire area
    sum_evap_pr = sum(df["e_atm_pr"].iloc[1:]) * dict_param["pr_no_meas_area"] / dict_param["tot_area"]
    sum_evap_cp = sum(df["e_atm_cp"].iloc[1:]) * dict_param["cp_no_meas_area"] / dict_param["tot_area"]
    sum_evap_op = sum(df["e_atm_op"].iloc[1:]) * dict_param["op_no_meas_area"] / dict_param["tot_area"]
    sum_evap_up = sum(df["e_atm_up"].iloc[1:]) * dict_param["up_no_meas_area"] / dict_param["tot_area"]
    sum_evap_uz = sum(df["t_atm_uz"].iloc[1:]) * dict_param["uz_no_meas_area"] / dict_param["tot_area"]
    sum_evap_ow = sum(df["e_atm_ow"].iloc[1:]) * dict_param["ow_no_meas_area"] / dict_param["tot_area"]
    sum_evap_meas = (sum(df["e_atm_meas"].iloc[1:]) * dict_param["tot_meas_area"] +
                     sum(df["t_atm_top_meas"].iloc[1:]) * dict_param["top_meas_area"] +
                     sum(df["t_atm_btm_meas"].iloc[1:]) * dict_param["btm_meas_area"]) / dict_param["tot_area"]
    sum_evap = sum_evap_pr + sum_evap_cp + sum_evap_op + sum_evap_up + sum_evap_uz + sum_evap_ow + sum_evap_meas

    # discharge to outside over entire area (added discharge from combined sewer system to outside on 03/03/2020 by Shiyang)
    sum_q_out = (sum(df["q_ow_out"].iloc[1:]) * dict_param["ow_no_meas_area"] +
                 sum(df["q_meas_out"].iloc[1:]) * dict_param["tot_meas_area"] +
                 sum(df["q_mss_out"].iloc[1:]) * dict_param["tot_mss_area"]) / dict_param["tot_area"]

    # seepage to deep groundwater over entire area
    sum_s_deepgw = sum(df["s_gw_out"].iloc[1:]) * dict_param["gw_no_meas_area"] / dict_param["tot_area"]

    # change in storages over entire area:
    # change in storages in pr, cp, op, up and uz
    sum_ds_pr = ((df["intstor_pr"].iloc[-1] - df["intstor_pr"].iloc[0]) * dict_param["pr_no_meas_area"] /
                 dict_param["tot_area"])
    sum_ds_cp = ((df["intstor_cp"].iloc[-1] - df["intstor_cp"].iloc[0]) * dict_param["cp_no_meas_area"] /
                 dict_param["tot_area"])
    sum_ds_op = ((df["intstor_op"].iloc[-1] - df["intstor_op"].iloc[0]) * dict_param["op_no_meas_area"] /
                 dict_param["tot_area"])
    sum_ds_up = ((df["fin_intstor_up"].iloc[-1] - df["fin_intstor_up"].iloc[0]) * dict_param["up_no_meas_area"] /
                 dict_param["tot_area"])
    sum_ds_uz = ((df["theta_uz"].iloc[-1] - df["theta_uz"].iloc[0]) * dict_param["uz_no_meas_area"] /
                 dict_param["tot_area"])

    # change in groundwater storage is a bit tricky to calculate
    storage_coef = df["sc_gw"]
    groundwater_level = df["gwl"]
    ds_gw = np.zeros_like(groundwater_level)
    for t in range(1, iters):
        ds_gw[t] = 1000 * storage_coef[t] * (groundwater_level[t-1] - groundwater_level[t])
    sum_ds_gw = sum(ds_gw) * dict_param["gw_no_meas_area"] / dict_param["tot_area"]
    sum_ds_gw_sl = (1000 * (df["gwl_sl"].iloc[-1] - df["gwl_sl"].iloc[0]) * dict_param["gw_no_meas_area"] /
                    dict_param["tot_area"])  # ? mark it here: after changing (old - new), still not working.

    # change in storage in sewer system
    sum_ds_swds = ((df["stor_swds"].iloc[-1] - df["stor_swds"].iloc[0]) * dict_param["swds_no_meas_area"] /
                   dict_param["tot_area"])
    sum_ds_mss = ((df["stor_mss"].iloc[-1] - df["stor_mss"].iloc[0]) * dict_param["mss_no_meas_area"] /
                  dict_param["tot_area"])
    sum_ds_ow = 1000 * (df["owl"].iloc[-1] - df["owl"].iloc[0]) * dict_param["ow_no_meas_area"] / dict_param["tot_area"]
    sum_ds_meas = (((df["intstor_meas"].iloc[-1] - df["intstor_meas"].iloc[0]) * dict_param["tot_meas_area"] +
                   (df["fin_stor_top_meas"].iloc[-1] - df["fin_stor_top_meas"].iloc[0]) * dict_param["top_meas_area"] +
                   (df["fin_stor_btm_meas"].iloc[-1] - df["fin_stor_btm_meas"].iloc[0]) * dict_param["btm_meas_area"]) /
                   dict_param["tot_area"])

    sum_ds = (sum_ds_pr + sum_ds_cp + sum_ds_op + sum_ds_up + sum_ds_uz + sum_ds_gw + sum_ds_gw_sl + sum_ds_swds +
              sum_ds_mss + sum_ds_ow + sum_ds_meas)

    # calculate the difference in water balance to check whether water balance is closed over entire area
    balance_diff = sum_prec - sum_evap - sum_q_out - sum_s_deepgw - sum_ds

    # statistics of entire model for logging
    stat_model = {"rain": round(sum_prec, 2), "evap": round(sum_evap, 2), "Q_out": round(sum_q_out, 2),
                  "seepage": round(sum_s_deepgw, 2), "storage diff":  round(sum_ds, 2),
                  "balance diff": balance_diff}
#    print("\n")
#    print("Entire model:")
    # display in console the table of water balance statistics over entire area
    headers = ["rain[mm]", "evap[mm]", "Q_out[mm]", "seepage[mm]", "storage diff[mm]", "balance diff[mm]"]
    table = [[round(sum_prec, 2), round(sum_evap, 2), round(sum_q_out, 2), round(sum_s_deepgw, 2), round(sum_ds, 2),
              balance_diff]]
#    print(tabulate(table, headers, tablefmt="presto"))

#    if math.isclose(balance_diff, 0, abs_tol=0.000001):
#        print("Water balance for entire model is closed.")
#    else:
#        warning_msg_model = "WARNING: Water balance for entire model is NOT closed. Please recheck!"
#        print(warning_msg_model)
#        warning_msgs.append(warning_msg_model)
#        # raise SystemExit("WARNING: Water balance is NOT closed for entire model. Please recheck.")

    # check water balance for measure itself:
    # precipitation over measure itself
    p_meas = sum(df["prec_meas"].iloc[1:])

    # evaporation and changes in storage over measure
    try:
        e_meas = sum_evap_meas * dict_param["tot_area"] / dict_param["tot_meas_area"]
        ds_meas = sum_ds_meas * dict_param["tot_area"] / dict_param["tot_meas_area"]
    except ZeroDivisionError:
        e_meas = 0
        ds_meas = 0

    # total runoff from inflow area to measure
    r_inflowarea_meas = sum(df["sum_r_meas"].iloc[1:])
    # groundwater recharge from measure
    gw_rech_meas = sum(df["q_meas_gw"].iloc[1:])
    # open water recharge from measure
    ow_rech_meas = sum(df["q_meas_ow"].iloc[1:])
    # discharge to SWDS from measure
    q_swds = sum(df["q_meas_swds"].iloc[1:])
    # discharge to MSS from measure
    q_mss = sum(df["q_meas_mss"].iloc[1:])
    # discharge to Outside from measure
    q_out = sum(df["q_meas_out"].iloc[1:])

    # calculate the difference in water balance to check whether water balance is closed over measure itself
    balance_diff_meas = p_meas - e_meas + r_inflowarea_meas - gw_rech_meas - ow_rech_meas - q_swds - q_mss - q_out - (
        ds_meas)

    # statistics of measure for logging
    stat_meas = {"rain": round(p_meas, 2), "evap": round(e_meas, 2), "inflow runoff": round(r_inflowarea_meas, 2),
                 "GW.rech": round(gw_rech_meas, 2), "OW.rech": round(ow_rech_meas, 2), "Q_swds": round(q_swds, 2),
                 "Q_mss": round(q_mss, 2), "Q_out": round(q_out, 2), "storage diff": round(ds_meas, 2),
                 "balance diff": balance_diff_meas}
#    print("\n")
#    print("Measure itself:")
    # display in console the table of water balance statistics for measure itself
    headers_m = ["rain[mm]", "evap[mm]", "inflow runoff[mm]", "OW.rech[mm]", "Q_swds[mm]", "Q_mss[mm]",
                 "Q_Out[mm]", "GW.rech[mm]", "storage diff[mm]", "balance diff[mm]"]
    table_m = [[round(p_meas, 2), round(e_meas, 2), round(r_inflowarea_meas, 2), round(ow_rech_meas, 2),
                round(q_swds, 2), round(q_mss, 2), round(gw_rech_meas, 2),
                round(q_out, 2), round(ds_meas, 2), balance_diff_meas]]
#    print(tabulate(table_m, headers_m, tablefmt="presto"))

#    if math.isclose(balance_diff_meas, 0, abs_tol=0.000001):
#        print("Water balance is closed for measure itself.")
#    else:
#        warning_msg_measure = "WARNING: Water balance for measure itself is NOT closed. Please recheck!"
#        print(warning_msg_measure)
#        warning_msgs.append(warning_msg_measure)
        # raise SystemExit("Water balance for measure is not closed. Please recheck.")

    # check water balance for measure inflow area:
    try:
        # precipitation over measure inflow area
        p_mia = sum_prec

        # evaporation from measure inflow area
        e_mia = ((sum(df["e_atm_meas"].iloc[1:]) * dict_param["tot_meas_area"] +
                 sum(df["t_atm_top_meas"].iloc[1:]) * dict_param["top_meas_area"] +
                 sum(df["t_atm_btm_meas"].iloc[1:] * dict_param["btm_meas_area"]) +
                 sum(df["e_atm_pr"].iloc[1:]) * (dict_param["pr_meas_inflow_area"] - dict_param["pr_meas_area"]) +
                 sum(df["e_atm_cp"].iloc[1:]) * (dict_param["cp_meas_inflow_area"] - dict_param["cp_meas_area"]) +
                 sum(df["e_atm_op"].iloc[1:]) * (dict_param["op_meas_inflow_area"] - dict_param["op_meas_area"]) +
                 sum(df["e_atm_up"].iloc[1:]) * (dict_param["up_meas_inflow_area"] - dict_param["up_meas_area"]) +
                 sum(df["e_atm_ow"].iloc[1:]) * (dict_param["ow_meas_inflow_area"] - dict_param["ow_meas_area"])) /
                 dict_param["tot_meas_inflow_area"])

        # change in storages from measure inflow area
        ds_mia = ((df["intstor_pr"].iloc[-1] - df["intstor_pr"].iloc[0]) *
                  (dict_param["pr_meas_inflow_area"] - dict_param["pr_meas_area"]) +
                  (df["intstor_cp"].iloc[-1] - df["intstor_cp"].iloc[0]) *
                  (dict_param["cp_meas_inflow_area"] - dict_param["cp_meas_area"]) +
                  (df["intstor_op"].iloc[-1] - df["intstor_op"].iloc[0]) *
                  (dict_param["op_meas_inflow_area"] - dict_param["op_meas_area"]) +
                  (df["fin_intstor_up"].iloc[-1] - df["fin_intstor_up"].iloc[0]) *
                  (dict_param["up_meas_inflow_area"] - dict_param["up_meas_area"]) +
                  1000 * (df["owl"].iloc[-1] - df["owl"].iloc[0]) *
                  (dict_param["ow_meas_inflow_area"] - dict_param["ow_meas_area"]) +
                  (df["intstor_meas"].iloc[-1] - df["intstor_meas"].iloc[0]) *
                  (dict_param["tot_meas_area"]) +
                  (df["fin_stor_top_meas"].iloc[-1] - df["fin_stor_top_meas"].iloc[0]) *
                  (dict_param["top_meas_area"]) +
                  (df["fin_stor_btm_meas"].iloc[-1] - df["fin_stor_btm_meas"].iloc[0]) *
                  (dict_param["btm_meas_area"])) / dict_param["tot_meas_inflow_area"]

        # open water recharge from measure inflow area
        ow_rech_meas_mia = ow_rech_meas * dict_param["tot_meas_area"] / dict_param["tot_meas_inflow_area"]
        # groundwater recharge from measure inflow area
        gw_rech_meas_mia = gw_rech_meas * dict_param["tot_meas_area"] / dict_param["tot_meas_inflow_area"]
        # discharge to SWDS from measure inflow area
        q_swds_meas_mia = q_swds * dict_param["tot_meas_area"] / dict_param["tot_meas_inflow_area"]
        # discharge to MSS from measure inflow area
        q_mss_meas_mia = q_mss * dict_param["tot_meas_area"] / dict_param["tot_meas_inflow_area"]
        # discharge to Outside from measure inflow area
        q_out_meas_mia = q_out * dict_param["tot_meas_area"] / dict_param["tot_meas_inflow_area"]

    except ZeroDivisionError:
        p_mia = e_mia = ds_mia = ow_rech_meas_mia = gw_rech_meas_mia = q_swds_meas_mia = q_mss_meas_mia = \
            q_out_meas_mia = 0.0

    # calculate the difference in water balance to check whether water balance is closed over measure inflow area
    balance_diff_mia = (p_mia - e_mia - ow_rech_meas_mia - gw_rech_meas_mia - q_swds_meas_mia - q_mss_meas_mia - ds_mia
                        - q_out_meas_mia)

    stat_mia = {"rain": round(p_mia, 2), "evap": round(e_mia, 2), "GW.rech": round(gw_rech_meas_mia, 2),
                "OW.rech": round(ow_rech_meas_mia), "Q_swds": round(q_swds_meas_mia, 2),
                "Q_mss": round(q_mss_meas_mia, 2), "Q_Out": round(q_out_meas_mia, 2),
                "storage diff": round(ds_mia, 2), "balance diff": balance_diff_mia}

#    print("\n")
#    print("Measure inflow area:")
    # display in console the table of water balance statistics for measure inflow area
    headers_mia = ["rain[mm]", "evap[mm]", "GW.rech[mm]", "OW.rech[mm]", "Q_swds[mm]", "Q_mss[mm]", "Q_Out[mm]",
                   "storage diff[mm]", "balance diff[mm]"]
    table_mia = [[round(p_mia, 2), round(e_mia, 2), round(gw_rech_meas_mia, 2), round(ow_rech_meas_mia, 2),
                  round(q_swds_meas_mia, 2), round(q_mss_meas_mia, 2), round(q_out_meas_mia, 2), round(ds_mia, 2),
                  balance_diff_mia]]
#    print(tabulate(table_mia, headers_mia, tablefmt="presto"))

#    if math.isclose(balance_diff_mia, 0, abs_tol=0.000001):
#        print("Water balance is closed for measure inflow area.")
#    else:
#        warning_msg_mia = "WARNING: Water balance for measure inflow area is NOT closed. Please recheck!"
#        print(warning_msg_mia)
#        warning_msgs.append(warning_msg_mia)
#        # raise SystemExit(warning_msg_mia)

    return stat_model, stat_meas, stat_mia, warning_msgs


def read_parameter_measure_csv(measure_id, parameter_base, apply_measure=True):
    """
    reads parameters from an Excel csv.

    Args:
        measure_id : id of the measure, required to obtain the correct parameters from the table
        parameter_base : a dictionary containing the catchment parameters, often read before this function is called
    Returns:
        (dictionary): A dictionary of parameters for measure.
    """
    path = Path.cwd() / ".." / "input"
    cf = pd.read_csv(str(path) + "\\" + 'Parameters measures.csv', index_col=0)
    
    # Check whether an exception measure is implemented or not. If so, take the first measure as placeholder for the measure parameters. These will not be implemented.
    try:
        int(measure_id)
        exception = False
    except:
        exception = True
    
    if exception:
        idx_meas = 0
    elif not exception:
        idx_meas = np.where(int(measure_id)==cf.index)[0][0]
    
    cf = cf.iloc[idx_meas]
    if cf['runoffcap_meas_soil_inherit']>0:
        cf['runoffcap_btm_meas'] = parameter_base['infilcap_up']
    
    if apply_measure:
        choice = True
    elif not apply_measure:
        choice = False
#    choice = cf["measure_applied"]
      
    if cf['cp_meas_inflow_area'] > 0 and cf['op_meas_inflow_area'] > 0:
        tot_meas_area = (parameter_base['tot_cp_area'] + parameter_base['tot_op_area']) / cf['Ain_def']
    elif cf['pr_meas_inflow_area'] > 0:
        tot_meas_area = parameter_base['tot_pr_area'] / cf['Ain_def']
    elif cf['cp_meas_inflow_area'] > 0:
        tot_meas_area = parameter_base['tot_cp_area'] / cf['Ain_def']
    elif cf['op_meas_inflow_area'] > 0:
        tot_meas_area = parameter_base['tot_op_area'] / cf['Ain_def']
    elif cf['up_meas_inflow_area'] > 0:
        tot_meas_area = parameter_base['tot_up_area'] / cf['Ain_def']
    elif cf['ow_meas_inflow_area'] > 0:
        tot_meas_area = parameter_base['tot_ow_area'] / cf['Ain_def']
    
    top_meas_area = btm_meas_area = tot_meas_area
    tot_meas_inflow_area = tot_meas_area * cf['Ain_def']
    
#    tot_meas_area = cf["tot_meas_area"]
#    top_meas_area = cf["top_meas_area"]
#    btm_meas_area = cf["btm_meas_area"]
#
#    pr_meas_inflow_area = cf["pr_meas_inflow_area"]
#    cp_meas_inflow_area = cf["cp_meas_inflow_area"]
#    op_meas_inflow_area = cf["op_meas_inflow_area"]
#    up_meas_inflow_area = cf["up_meas_inflow_area"]
#    ow_meas_inflow_area = cf["ow_meas_inflow_area"]
#    tot_meas_inflow_area = pr_meas_inflow_area + cp_meas_inflow_area + op_meas_inflow_area + up_meas_inflow_area + \
#                           ow_meas_inflow_area
    # deal with initial values here later
    validinput = False
    while not validinput:
        if not choice:  # input choice: no measure
            # to make it as foolproof as possible, when measure is not applied, measure-related area will all be set as
            # zeros regardless of what is in the configuration file.

            tot_meas_area = pr_meas_area = (
                cp_meas_area
            ) = (
                op_meas_area
            ) = (
                up_meas_area
            ) = (
                uz_meas_area
            ) = gw_meas_area = swds_meas_area = mss_meas_area = ow_meas_area = top_meas_area = btm_meas_area = 0.0

            pr_meas_inflow_area = cp_meas_inflow_area = op_meas_inflow_area = up_meas_inflow_area = ow_meas_inflow_area \
                = tot_meas_inflow_area = 0.0

            validinput = True
        elif choice:  # input choice: there is measure
            pr_meas_area = (cf["pr_meas_area"]>0) * tot_meas_area
            cp_meas_area = (cf["cp_meas_area"]>0) * tot_meas_area
            op_meas_area = (cf["op_meas_area"]>0) * tot_meas_area
            up_meas_area = (cf["up_meas_area"]>0) * tot_meas_area
            uz_meas_area = (cf["uz_meas_area"]>0) * tot_meas_area
            gw_meas_area = (cf["gw_meas_area"]>0) * tot_meas_area
            swds_meas_area = (cf["swds_meas_area"]>0) * tot_meas_area
            mss_meas_area = (cf["mss_meas_area"]>0) * tot_meas_area
            ow_meas_area = (cf["ow_meas_area"]>0) * tot_meas_area
            
            if cf['cp_meas_area']>0 and cf['op_meas_area']>0:
                cp_meas_area = parameter_base['tot_cp_area'] / cf['Ain_def']
                op_meas_area = parameter_base['tot_op_area'] / cf['Ain_def']
            
            pr_meas_inflow_area = pr_meas_area * cf['Ain_def']
            cp_meas_inflow_area = cp_meas_area * cf['Ain_def']
            op_meas_inflow_area = op_meas_area * cf['Ain_def']
            up_meas_inflow_area = up_meas_area * cf['Ain_def']
            ow_meas_inflow_area = ow_meas_area * cf['Ain_def']
            
            validinput = True
        else:
            raise ValueError("Error: Choice can only be true or false.")
    # these parameters are parameters for measure. As you can see in the configuration file,
    # these so many parameters are confusing and overview of parameters is not as good as excel.
    # Hence, it may be possible that we build a GUI to handle this problem. But for the time being,
    # we just build like this to make it run first.

    # tot_meas_area -- predefined measure area [m^2]
    # Button_BW17 --- predefined selection at which measure layer runoff from other areas is stored (1 or 3), Inflow from other areas can only take place at interception level (1) or at the bottom storage level (3).
    # * intstor_meas_prevt --- interception storage on the measure at previous time step [mm]
    # intstor_meas_t0 --- predefined interception storage on the measure at t=0 [mm]
    # EV_evaporation --- predefined selection if evaporation from measure is possible (1) or not (0)
    # num_stor_lvl --- predefined number of storage levels (1, 2 or 3)
    # infilcap_int_meas --- predefined infiltration capacity of measure [mm/d] (4800mm/d)
    # storcap_top_meas --- predefined storage capacity in top layer of measure (76.2mm)
    # storcap_btm_meas --- predefined storage capacity in bottom layer of measure (182.88mm)
    # * stor_top_meas_prevt --- top layer storage at the end of previous time step [mm]
    # top_stor_meas_t0 --- top layer storage at t = 0 [mm] (0 mm)
    # * stor_btm_meas_prevt --- bottom layer storage at the end of previous time step [mm]
    # bot_stor_meas_t0 --- bottom layer storage at t = 0 [mm] (0 mm)

    # storcap_int_meas --- predefined interception storage capacity of measure [mm] (20mm)
    # top_meas_area --- predefined area of top layer storage area of measure [m^2]
    # ET_transpiration --- predefined selection if transpiration from measure is possible (1) or not (0)
    # evaporation_factor_meas --- predefined evaporation factor of measure [-]
    # infilcap_top_meas --- predefined infiltration capacity of top layer of measure [mm/d] (480mm/d)

    # btm_meas_area --- predefined area of bottom layer storage area of measure [m^2]
    # btm_meas_transpiration --- predefined selection if transpiration from bottom layer of measure is possible (1) or not (0)
    # connection_to_gw --- predefined selection if percolation (connection) from measure to groundwater is possible (1) or not (0)
    # limited_by_gwl --- predefined limitation of percolation from measure to groundwater if groundwater level is below measure bottom level (1=yes; 0=no)
    # btm_level_meas --- predefined bottom level of measure [m -SL] (0.6858)
    # btm_discharge_type --- predefined definition of discharge type from bottom layer of measure (0 = down_seepage_flux limited, 1 = level difference over resistance)
    # runoffcap_btm_meas --- predefined runoff capacity from bottom layer of measure [mm/d] (down_seepage_flux=15mm/d)
    # dischlvl_btm_meas --- predefined discharge level from bottom layer of measure [mm]
    # c_btm_meas --- predefined hydraulic resistance for level induced discharge from bottom layer of measure [d]

    # surf_runoff_meas_OW --- predefined definition of surface runoff from measure to open water (0 = no, 1 = yes)
    # ctrl_runoff_meas_OW --- predefined definition of controlled runoff from measure to open water (0 = no, 1 = yes)
    # overflow_meas_OW --- predefined definition of overflow from measure to open water (0 = no, 1 = yes)
    # surf_runoff_meas_UZ --- predefined definition of surface runoff from measure to unsaturated zone (0 = no, 1 = yes)
    # ctrl_runoff_meas_UZ --- predefined definition of controlled runoff from measure to unsaturated zone (0 = no, 1 = yes)
    # overflow_meas_UZ --- predefined definition of overflow from measure to unsaturated zone (0 = no, 1 = yes)
    # surf_runoff_meas_GW --- predefined definition of surface runoff from measure to groundwater (0 = no, 1 = yes)
    # ctrl_runoff_meas_GW --- predefined definition of controlled runoff from measure to groundwater (0 = no, 1 = yes)
    # overflow_meas_GW --- predefined definition of overflow from measure to groundwater (0 = no, 1 = yes)
    # surf_runoff_meas_SWDS --- predefined definition of surface runoff from measure to storm water drainage system (0 = no, 1 = yes)
    # ctrl_runoff_meas_SWDS --- predefined definition of controlled runoff from measure to storm water drainage system (0 = no, 1 = yes)
    # overflow_meas_GW --- predefined definition of overflow from measure to storm water drainage system (0 = no, 1 = yes)
    # surf_runoff_meas_MSS --- predefined definition of surface runoff from measure to mixed sewer system (0 = no, 1 = yes)
    # ctrl_runoff_meas_MSS --- predefined definition of controlled runoff from measure to mixed sewer system (0 = no, 1 = yes)
    # overflow_meas_GW--- predefined definition of overflow from measure to mixed sewer system (0 = no, 1 = yes)
    # surf_runoff_meas_Out --- predefined definition of surface runoff from measure to outside water (0 = no, 1 = yes)
    # ctrl_runoff_meas_Out --- predefined definition of controlled runoff from measure to outside water (0 = no, 1 = yes)
    # overflow_meas_Out --- predefined definition of overflow from measure to outside water (0 = no, 1 = yes)

    runoff_to_stor_layer = cf["runoff_to_stor_layer"]
    intstor_meas_t0 = cf["intstor_meas_t0"]
    EV_evaporation = cf["EV_evaporation"]
    num_stor_lvl = cf["num_stor_lvl"]
    infilcap_int_meas = cf["infilcap_int_meas"]
    storcap_top_meas = cf["storcap_top_meas"]
    storcap_btm_meas = cf["storcap_btm_meas"]
    stor_top_meas_t0 = cf["stor_top_meas_t0"]
    stor_btm_meas_t0 = cf["stor_btm_meas_t0"]
    storcap_int_meas = cf["storcap_int_meas"]
    ET_transpiration = cf["ET_transpiration"]
    evaporation_factor_meas = cf["evaporation_factor_meas"]
    IN_infiltration = cf["IN_infiltration"]
    infilcap_top_meas = cf["infilcap_top_meas"]
    btm_meas_transpiration = cf["btm_meas_transpiration"]
    connection_to_gw = cf["connection_to_gw"]
    limited_by_gwl = cf["limited_by_gwl"]
    btm_level_meas = cf["btm_level_meas"]
    btm_discharge_type = cf["btm_discharge_type"]
    runoffcap_btm_meas = cf["runoffcap_btm_meas"]
    dischlvl_btm_meas = cf["dischlvl_btm_meas"]
    c_btm_meas = cf["c_btm_meas"]

    # Buttons:
    surf_runoff_meas_OW = cf["surf_runoff_meas_OW"]
    ctrl_runoff_meas_OW = cf["ctrl_runoff_meas_OW"]
    overflow_meas_OW = cf["overflow_meas_OW"]
    surf_runoff_meas_UZ = cf["surf_runoff_meas_UZ"]
    ctrl_runoff_meas_UZ = cf["ctrl_runoff_meas_UZ"]
    overflow_meas_UZ = cf["overflow_meas_UZ"]
    surf_runoff_meas_GW = cf["surf_runoff_meas_GW"]
    ctrl_runoff_meas_GW = cf["ctrl_runoff_meas_GW"]
    overflow_meas_GW = cf["overflow_meas_GW"]
    surf_runoff_meas_SWDS = cf["surf_runoff_meas_SWDS"]
    ctrl_runoff_meas_SWDS = cf["ctrl_runoff_meas_SWDS"]
    overflow_meas_SWDS = cf["overflow_meas_SWDS"]
    surf_runoff_meas_MSS = cf["surf_runoff_meas_MSS"]
    ctrl_runoff_meas_MSS = cf["ctrl_runoff_meas_MSS"]
    overflow_meas_MSS = cf["overflow_meas_MSS"]
    surf_runoff_meas_Out = cf["surf_runoff_meas_Out"]
    ctrl_runoff_meas_Out = cf["ctrl_runoff_meas_Out"]
    overflow_meas_Out = cf["overflow_meas_Out"]

    # Note that pr_meas_inflow_area should be within the range (pr_meas_area, tot_pr_area), it should be specified.
    # But for the time being we assume it is equal to pr_meas_area.
    # Assume for the time being, measure inflow area = component_area

    if tot_meas_area != (pr_meas_area + cp_meas_area+ op_meas_area + up_meas_area + uz_meas_area + gw_meas_area
        + swds_meas_area + mss_meas_area + ow_meas_area):
        raise ValueError("Error: Measure area info error")
    greenroof_type_measure = cf["greenroof_type_measure"]
    # 0 or 1 check. check some parameters (some buttons) which can only be selected from 0 or 1
    k = [surf_runoff_meas_OW, ctrl_runoff_meas_OW, overflow_meas_OW, surf_runoff_meas_UZ, ctrl_runoff_meas_UZ, overflow_meas_UZ, surf_runoff_meas_GW, ctrl_runoff_meas_GW, overflow_meas_GW,
         surf_runoff_meas_SWDS, ctrl_runoff_meas_SWDS, overflow_meas_SWDS, surf_runoff_meas_MSS, ctrl_runoff_meas_MSS, overflow_meas_MSS, surf_runoff_meas_Out, ctrl_runoff_meas_Out, overflow_meas_Out,
         limited_by_gwl, connection_to_gw, btm_meas_transpiration, btm_discharge_type]
    check = [n for n in k if n != 0 and n != 1]
    if len(check) != 0:
        print(check)
        raise ValueError("Error: Button Parameter can only be 0 or 1.")
    if num_stor_lvl != 1 and num_stor_lvl != 2 and num_stor_lvl != 3:
        # print(num_stor_lvl)
        raise ValueError("Error: Number of storage levels can only be (1, 2 or 3) (integer)")
    if runoff_to_stor_layer !=1 and runoff_to_stor_layer != 3:
        raise ValueError("Error: runoff_to_stor_layer (Runoff from other areas into storage layer) can only be (1 or 3)")

    title = cf["title"]
    return {
        "pr_meas_area": pr_meas_area,
        "cp_meas_area": cp_meas_area,
        "op_meas_area": op_meas_area,
        "up_meas_area": up_meas_area,
        "uz_meas_area": uz_meas_area,
        "gw_meas_area": gw_meas_area,
        "swds_meas_area": swds_meas_area,
        "mss_meas_area": mss_meas_area,
        "ow_meas_area": ow_meas_area,
        "tot_meas_area": tot_meas_area,
        "pr_meas_inflow_area": pr_meas_inflow_area,
        "cp_meas_inflow_area": cp_meas_inflow_area,
        "op_meas_inflow_area": op_meas_inflow_area,
        "up_meas_inflow_area": up_meas_inflow_area,
        "ow_meas_inflow_area": ow_meas_inflow_area,
        "runoff_to_stor_layer": runoff_to_stor_layer,
        "intstor_meas_t0": intstor_meas_t0,
        "EV_evaporation": EV_evaporation,
        "num_stor_lvl": num_stor_lvl,
        "infilcap_int_meas": infilcap_int_meas,
        "storcap_top_meas": storcap_top_meas,
        "storcap_btm_meas": storcap_btm_meas,
        "stor_top_meas_t0": stor_top_meas_t0,
        "stor_btm_meas_t0": stor_btm_meas_t0,
        "storcap_int_meas": storcap_int_meas,
        "top_meas_area": top_meas_area,
        "ET_transpiration": ET_transpiration,
        "evaporation_factor_meas": evaporation_factor_meas,
        "IN_infiltration": IN_infiltration,
        "infilcap_top_meas": infilcap_top_meas,
        "btm_meas_area": btm_meas_area,
        "btm_meas_transpiration": btm_meas_transpiration,
        "connection_to_gw": connection_to_gw,
        "limited_by_gwl": limited_by_gwl,
        "btm_level_meas": btm_level_meas,
        "btm_discharge_type": btm_discharge_type,
        "runoffcap_btm_meas": runoffcap_btm_meas,
        "dischlvl_btm_meas": dischlvl_btm_meas,
        "c_btm_meas": c_btm_meas,
        "surf_runoff_meas_OW": surf_runoff_meas_OW,
        "ctrl_runoff_meas_OW": ctrl_runoff_meas_OW,
        "overflow_meas_OW": overflow_meas_OW,
        "surf_runoff_meas_UZ": surf_runoff_meas_UZ,
        "ctrl_runoff_meas_UZ": ctrl_runoff_meas_UZ,
        "overflow_meas_UZ": overflow_meas_UZ,
        "surf_runoff_meas_GW": surf_runoff_meas_GW,
        "ctrl_runoff_meas_GW": ctrl_runoff_meas_GW,
        "overflow_meas_GW": overflow_meas_GW,
        "surf_runoff_meas_SWDS": surf_runoff_meas_SWDS,
        "ctrl_runoff_meas_SWDS": ctrl_runoff_meas_SWDS,
        "overflow_meas_SWDS": overflow_meas_SWDS,
        "surf_runoff_meas_MSS": surf_runoff_meas_MSS,
        "ctrl_runoff_meas_MSS": ctrl_runoff_meas_MSS,
        "overflow_meas_MSS": overflow_meas_MSS,
        "surf_runoff_meas_Out": surf_runoff_meas_Out,
        "ctrl_runoff_meas_Out": ctrl_runoff_meas_Out,
        "overflow_meas_Out": overflow_meas_Out,
        "greenroof_type_measure": greenroof_type_measure,
        "tot_meas_inflow_area": tot_meas_inflow_area,  # note
        "title": title
    }
    

def read_parameter_base_dic(dictionary):
    """
    reads parameters from the TOML-formated static form.

    Args:
        dictionary: a dictionary containing the catchment properties
        
    Returns:
        (dictionary): A dictionary of general parameters
    """
    cf = dictionary
    delta_t = cf["timestep"] / 86400  # length of timestep, converted from second (s) to day (d)
    tot_area = cf["tot_area"]  # total area of study area (model) [m^2]
    soiltype = cf["soiltype"]  # soil type
    croptype = cf["croptype"]  # crop type
    area_type = cf["area_type"]  # area input type [0: fraction, 1: area]
    validinput = False
    while not validinput:
        if area_type == 0:  # input area type: fraction
            pr_frac = cf["pr_frac"]  # paved roof fraction of total [-]
            cp_frac = cf["cp_frac"]  # closed paved fraction of total [-]
            op_frac = cf["op_frac"]  # open paved fraction of total [-]
            up_frac = cf["up_frac"]  # unpaved fraction of total [-]
            ow_frac = cf["ow_frac"]  # open water fraction of total [-]
            tot_pr_area = pr_frac * tot_area  # total area of paved roof [m^2]
            frac_pr_aboveGW = cf["frac_pr_aboveGW"]  # part of buildings (PR) above GW [-]
            tot_cp_area = cp_frac * tot_area  # total area of closed paved [m^2]
            tot_op_area = op_frac * tot_area  # total area of open paved [m^2]
            tot_up_area = up_frac * tot_area  # total area of unpaved [m^2]
            tot_ow_area = ow_frac * tot_area  # total area of open water [m^2]
            frac_ow_aboveGW = cf["frac_ow_aboveGW"]  # part of open water above GW [-]
            tot_uz_area = tot_up_area  # total area of unsaturated zone (Assumed equal to total area of unpaved) [m^2]
            gw_frac = (
                pr_frac * frac_pr_aboveGW
                + cp_frac
                + op_frac
                + up_frac
                + ow_frac * frac_ow_aboveGW
            )  # groundwater fraction of total [-]
            tot_gw_area = gw_frac * tot_area  # total area of groundwater [m^2]
            if math.isclose(
                pr_frac + cp_frac + op_frac + up_frac + ow_frac,
                1,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                validinput = True
            else:
                raise ValueError("Error: Land use fractions do not add up to 1.")
        elif area_type == 1:  # input area type: area
            tot_pr_area = cf["tot_pr_area"]
            pr_frac = tot_pr_area / tot_area
            frac_pr_aboveGW = cf["frac_pr_aboveGW"]
            tot_cp_area = cf["tot_cp_area"]
            cp_frac = tot_cp_area / tot_area
            tot_op_area = cf["tot_op_area"]
            op_frac = tot_op_area / tot_area
            tot_up_area = cf["tot_up_area"]
            up_frac = tot_up_area / tot_area
            tot_ow_area = cf["tot_ow_area"]
            ow_frac = tot_ow_area / tot_area
            frac_ow_aboveGW = cf["frac_ow_aboveGW"]
            tot_uz_area = tot_up_area
            gw_frac = (
                pr_frac * frac_pr_aboveGW
                + cp_frac
                + op_frac
                + up_frac
                + ow_frac * frac_ow_aboveGW
            )
            tot_gw_area = gw_frac * tot_area
            if math.isclose(
                tot_pr_area + tot_cp_area + tot_op_area + tot_up_area + tot_ow_area,
                tot_area,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                validinput = True
            else:
                raise ValueError(
                    "Error: land use areas do not add up to total area."
                )
        else:
            raise ValueError("Error: Input area type can only be 0-fraction or 1-area.")

    # part of paved area disconnected from sewer system
    discfrac_pr = cf[
        "discfrac_pr"
    ]  # part of paved roof disconnected from sewer system [-]
    discfrac_cp = cf[
        "discfrac_cp"
    ]  # part of closed paved disconnected from sewer system [-]
    discfrac_op = cf[
        "discfrac_op"
    ]  # part of open paved disconnected from sewer system [-]

    # interception storage capacity
    intstorcap_pr = cf[
        "intstorcap_pr"
    ]  # interception storage capacity on paved roof [mm]
    intstorcap_cp = cf[
        "intstorcap_cp"
    ]  # interception storage capacity on closed paved [mm]
    intstorcap_op = cf[
        "intstorcap_op"
    ]  # interception storage capacity on open paved [mm]
    intstorcap_up = cf["intstorcap_up"]  # interception storage capacity on unpaved [mm]
    storcap_ow = cf["storcap_ow"]  # storage capacity of open water [mm]

    # infiltration capacity
    infilcap_op = cf[
        "infilcap_op"
    ]  # infiltration capacity of open paved [mm/d]
    infilcap_up = cf["infilcap_up"]  # infiltration capacity of unpaved [mm/d]

    # rainfall statistics
    rainfall_swds_so = cf[
        "rainfall_swds_so"
    ]  # rainfall intensity when sewer overflow occurs on the street, in NL, it is T=2yr rainfall intensity [mm/dt]
    rainfall_mss_ow = cf[
        "rainfall_mss_ow"
    ]  # rainfall intensity when combined sewer overflow to open water occurs, in NL, it is T=1/6yr rainfall intensity [mm/dt]

    # sewer system parameters
    swds_frac = cf["swds_frac"]  # storm water drainage system fraction [-]
    mss_frac = 1.0 - swds_frac  # combined sewer system fraction [-]
    tot_disc_area = (
        tot_pr_area * discfrac_pr
        + tot_cp_area * discfrac_cp
        + tot_op_area * discfrac_op
    )
    tot_swds_area = swds_frac * (
        tot_pr_area + tot_cp_area + tot_op_area - tot_disc_area
    )
    tot_mss_area = mss_frac * (tot_pr_area + tot_cp_area + tot_op_area - tot_disc_area)
    storcap_swds = cf[
        "storcap_swds"
    ]  # storage capacity of storm water drainage system [mm]
    storcap_mss = cf["storcap_mss"]  # storage capacity of combined sewer system [mm]
    # discharge capacity of SWDS to open water [mm/dt], i.e. sewer discharge capacity of SWDS above which sewer overflow
    # onto street occurs, in NL, the design standard is it occurs once every two year.
    q_swds_ow_cap = (
        rainfall_swds_so - intstorcap_cp - storcap_swds
    )
    # discharge capacity of MSS to open water [mm/dt], i.e. sewer discharge capacity of MSS above which sewer overflow
    # onto street occurs, in NL, the design standard is it occurs once every two year.
    q_mss_ow_cap = (
        rainfall_swds_so - intstorcap_cp - storcap_mss
    )
    # discharge capacity of MSS to WWTP [mm/dt], i.e. sewer discharge capacity of MSS above which combined sewer
    # overflow to open water through CSO weir occurs, in NL, the design standard is it occurs six times per year.
    q_mss_out_cap = (
        rainfall_mss_ow - intstorcap_cp
    )

    # groundwater parameters
    w = cf["w"]  # drainage resistance w from groundwater to open water [d]
    seepage_define = cf["seepage_define"]  # defined seepage [0: flux, 1: level]
    if seepage_define == 0 or seepage_define == 1:
        down_seepage_flux = cf[
            "down_seepage_flux"
        ]  # constant downward flux from shallow groundwater to deep groundwater [mm/d] (negative means upward)
        gwl_t0 = cf["gwl_t0"]
        head_deep_gw = cf["head_deep_gw"]  # hydraulic head of deep groundwater [m-SL]
        vc = cf["vc"]  # vertical flow resistance from shallow groundwater to deep groundwater vc [d]
    else:
        raise ValueError(
            "Error: Seepage to deep groundwater can only be defined as either 0-flux or 1-level."
        )

    # open water parameters.
    q_ow_out_cap = (cf["q_ow_out_cap"])  # discharge capacity from open water to outside water over entire area [mm/d]
    ow_level = (
        storcap_ow / 1000.0
    )  # predefined target open water level, also initial open water level (at t=0) [m-Sl]

    # Non-negative check
    list1 = [
        delta_t,
        tot_area,
        soiltype,
        croptype,
        area_type,
        tot_pr_area,
        tot_cp_area,
        tot_op_area,
        tot_up_area,
        tot_ow_area,
        tot_uz_area,
        tot_gw_area,
        frac_pr_aboveGW,
        frac_ow_aboveGW,
        discfrac_pr,
        discfrac_cp,
        discfrac_op,
        intstorcap_pr,
        intstorcap_cp,
        intstorcap_op,
        intstorcap_up,
        ow_level,
        infilcap_op,
        infilcap_up,
        swds_frac,
        storcap_swds,
        storcap_mss,
        rainfall_swds_so,
        rainfall_mss_ow,
        q_swds_ow_cap,
        q_mss_ow_cap,
        q_mss_out_cap,
        w,
        seepage_define,
        gwl_t0,
        head_deep_gw,
        vc,
        q_ow_out_cap,
        ow_level,
    ]  # note that: down_seepage_flux can be negative(when upward down_seepage_flux)

    # Fraction within [0,1] check
    list2 = [
        pr_frac,
        cp_frac,
        op_frac,
        up_frac,
        ow_frac,
        gw_frac,
        frac_pr_aboveGW,
        frac_ow_aboveGW,
        discfrac_pr,
        discfrac_cp,
        discfrac_op,
        swds_frac,
    ]

    k1 = [n for n in list1 if n < 0]
    k2 = [n for n in list2 if n > 1 or n < 0]
    if len(k1) != 0:
        print(k1)
        raise ValueError("Error: Parameter is negative.")
    if len(k2) != 0:
        print(k2)
        raise ValueError("Error: Fraction is over 1 or negative.")

    intstor_pr_t0 = cf["intstor_pr_t0"]
    intstor_cp_t0 = cf["intstor_cp_t0"]
    intstor_op_t0 = cf["intstor_op_t0"]
    fin_intstor_up_t0 = cf["fin_intstor_up_t0"]
    stor_swds_t0 = cf["stor_swds_t0"]
    so_swds_t0 = cf["so_swds_t0"]
    stor_mss_t0 = cf["stor_mss_t0"]
    so_mss_t0 = cf["so_mss_t0"]

    return {
        "delta_t": delta_t,
        "tot_area": tot_area,
        "soiltype": soiltype,
        "croptype": croptype,
        "tot_pr_area": tot_pr_area,
        "tot_cp_area": tot_cp_area,
        "tot_op_area": tot_op_area,
        "tot_up_area": tot_up_area,
        "tot_ow_area": tot_ow_area,
        "tot_uz_area": tot_uz_area,
        "tot_gw_area": tot_gw_area,
        "discfrac_pr": discfrac_pr,
        "discfrac_cp": discfrac_cp,
        "discfrac_op": discfrac_op,
        "swds_frac": swds_frac,
        "tot_swds_area": tot_swds_area,
        "tot_mss_area": tot_mss_area,
        "storcap_swds": storcap_swds,
        "storcap_mss": storcap_mss,
        "intstorcap_pr": intstorcap_pr,
        "intstorcap_cp": intstorcap_cp,
        "intstorcap_op": intstorcap_op,
        "intstorcap_up": intstorcap_up,
        "infilcap_op": infilcap_op,
        "infilcap_up": infilcap_up,
        "w": w,
        "seepage_define": seepage_define,
        "down_seepage_flux": down_seepage_flux,
        "gwl_t0": gwl_t0,
        "head_deep_gw": head_deep_gw,
        "vc": vc,
        "q_swds_ow_cap": q_swds_ow_cap,
        "q_mss_ow_cap": q_mss_ow_cap,
        "q_mss_out_cap": q_mss_out_cap,
        "q_ow_out_cap": q_ow_out_cap,
        "ow_level": ow_level,
        "intstor_pr_t0": intstor_pr_t0,
        "intstor_cp_t0": intstor_cp_t0,
        "intstor_op_t0": intstor_op_t0,
        "fin_intstor_up_t0": fin_intstor_up_t0,
        "stor_swds_t0": stor_swds_t0,
        "so_swds_t0": so_swds_t0,
        "stor_mss_t0": stor_mss_t0,
        "so_mss_t0": so_mss_t0,
    }
    
    
# =============================================================================
# Following part is all for the function 'getconstants'
#
# Edited: Added an 'if' that when the runoff reduction factor exceeds the 1000,
# it will give give 1000. 
# =============================================================================

def making_marks(precipitation):
    """
    Make the marks by separating rainfall events by six consecutive hours without precipitation

    Args:
        precipitation (series): a series ("P_atm" column) of the dataframe

    Return:
        (numpy.ndarray): an array of corresponding marks for separating precipitation time series
    """
    # Create an empty array.
    mark = np.zeros_like(precipitation)
    # Specify values to this mark array.
    for i in range(len(precipitation)):
        if i < 6:
            mark[i] = 0
        else:
            if precipitation[i] > 0:
                if sum(precipitation[i-6:i]) > 0:
                    mark[i] = mark[i-1]
                else:
                    mark[i] = mark[i-1] + 1
            else:
                mark[i] = mark[i-1]
    return mark


def ranking(df, x, num):
    """
    According to the event mark, get the sum of x for each event, and then rank the sum from highest to lowest.

    Args:
        df (dataframe): a dataframe to do computations on
        x (string): a header of the dataframe
        num (integer): the total number of events

    Returns:
        (numpy.ndarray): an array of values ranked in a descending order
    """
    rank = np.zeros(num)
    for i in range(num):
        rank[i] = sum(df[df.mark == i][x])
    return sorted(rank, reverse=True)


def removekey(d, *keys):
    """
    Remove keys in the dictionary

    Args:
        d (dictionary): a dictionary to be modified
        keys (string): keys in the dictionary to be removed

    Returns:
        (dictionary): a modified dictionary
    """
    r = dict(d)
    for _ in keys:
        del r[_]
    return r


def find_corresponding_T_for_array(t_array, array, vararr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]):
    """
    Compute corresponding return period T (i.e. T=1/P, P is the probability of exceedance) for a certain return value in
    an array through linear interpolation, in order to compute an averaged value as runoff frequency reduction factor
    (The algorithm can be modified with the new code in the jupyter notebook despite the same results)

    Args:
        t_array ()
    Returns:
    """
    database = []
    for var in vararr:
        # print(var, 'case:')
        t_value = 0.0
        try:
            for counter, value in enumerate(array):
                if value < var:
                    # print(value)
                    v_below = array[counter]
                    v_above = array[counter-1]
                    # print('v-above', counter-1, v_above)
                    # print('v-below', counter, v_below)
                    # print('---'*6)
                    t_up = t_array[counter-1]
                    t_below = t_array[counter]
                    # print('T-up', t_up)
                    # print('T-below', t_below)
                    t_value = t_up - (v_above - var)/(v_above - v_below) * (t_up - t_below)
                    # print('T_value', t_value)
                    break
        except KeyError:
            # print('below',counter, array[counter])
            # print('above',counter, array[counter])
            t_value = math.inf
        finally:
            database.append(t_value)
    return database


def getconstants_measures(data, num_year=30):
    """
    Get the constant --- Runoff frequency reduction factor averaged over several specified runoff return value.

    Args:
        inputfilename (string): filename of the runoff time series resulted from the urbanwb model
        num_year (integer): total number of years of the time series
    """
    m = Analyse(data, num_year=num_year)
    results = m.getconstants()
    mean_constants = []
    for key in results.keys():
        new_var_array = []
        var_array = results[key]
        for var in var_array:
            if var < 2000:
                new_var_array.append(var)
        if new_var_array is not None:
            mean_constants.append(np.round(np.mean(new_var_array), 2))
    for i in range(len(mean_constants)):
        if np.isnan(mean_constants[i]) == True or mean_constants[i] > 1000:
            mean_constants[i] = 1000
        else:
            pass
    else:
        pass
    
    # if there is no change in runoff, then reduction factor = 0 (e.g. at implementing on unpaved when he unpaved area already has no runoff)
    if data[data.keys()[3]].sum() == data['Baseline'].sum():
        mean_constants = [1]        
    return results, mean_constants


class Analyse(object):
    """
    Integrate all functions, basically functioning, requiring further development
    """
    def __init__(self, data, num_year=30, ):
        self.output_name = "results_measures.csv"
        self.df = data
        self.df = self.df.fillna(0)
        self.dictionary = self.df.to_dict('list')
        self.num_year = num_year

        # making event marks according to precipitation (6 consective zeros as separation)
        self.df["mark"] = making_marks(self.df["P_atm"])
        self.measure_dictionary = removekey(self.dictionary, "Date", "P_atm", "Baseline")
        self.makingranks = self.makingranks()

    def getconstants(self,):  # consider changing function name to avoid confusion.
        pass
#        print(["storage cap mm", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50])
        emp = dict()
        baseT = find_corresponding_T_for_array(t_array=self.makingranks["T_list"], array=self.makingranks["Rank_baseline"])
        for key in self.makingranks.keys():
            if key not in ["Rank_P", "T_list", "Rank_baseline"]:
                a = find_corresponding_T_for_array(t_array=self.makingranks["T_list"], array=self.makingranks[key])
                c = [y/x for x, y in zip(baseT, a)]
                emp[key] = c
                np.mean(c)
        return emp

    def save_constants(self):
        pass

    def makingranks(self, ):
        # unchanged, I made a mistake here, should be self.emp rather than emp. Not a big problem.
        emp = dict()
        emp["Rank_P"] = ranking(self.df, "P_atm", int(max(self.df.mark) + 1))
        # create T list (30 yr, thus starting from (30+1/1) according to Weibull formula)
        emp["T_list"] = [(self.num_year + 1) / m for m in range(1, len(emp["Rank_P"]) + 1)]
        # rank runoff on the baseline case
        emp["Rank_baseline"] = ranking(self.df, "Baseline", int(max(self.df.mark) + 1))
        for key in self.measure_dictionary.keys():
            emp[key] = ranking(self.df, key, int(max(self.df.mark) + 1))
        data = pd.DataFrame.from_dict(emp)
        return data

    def save_to_csv(self, ):
        self.makingranks.to_csv(self.output_name)

    def plotting(self, measure_name, addition_name, xlim_down=0, xlim_up=40, ):
        self.data = self.makingranks

        plt.figure(figsize=(9, 6))
        plt.semilogy(self.data.Rank_P, self.data.T_list, "b--", label="Precipitation", ms=2)
        plt.semilogy(self.data.Rank_baseline, self.data.T_list, "k-", label="Baseline", ms=2)
        measures_rank_dictionary = removekey(self.data.to_dict('list'), "Rank_P", "Rank_baseline", "T_list")

        for key in measures_rank_dictionary.keys():
            plt.semilogy(measures_rank_dictionary[key], self.data.T_list, label=key, ms=2)

        x = np.linspace(0, 100, 200)
        # plt.legend(loc='best',frameon=False)
        plt.legend(loc='upper right', frameon=True)
        plt.xlabel("Runoff (mm)")
        plt.ylabel("T (year)")
        plt.title(measure_name + "(1981-2011)")
        plt.xlim(xlim_down, xlim_up)

        # add grid
        ax = plt.gca()
        ax.yaxis.grid(linestyle='--', linewidth=0.5, which='both')
        ax.xaxis.grid(linestyle='--', linewidth=0.5, which='both')

        #plt.savefig("figures/" + addition_name + measure_name + ".png")
        
# =============================================================================
# The following functions are edited from 'uwbmb_functions.py'
#
# Mainly, the functions have been edited to read .csv files instead of .ini.
# This allows for the user to give a list of measures as input in one overview.
# =============================================================================

def read_parameters_csv(stat1_inp, measure_id, neighbourhood_id, apply_measure=True):
    """
    reads parameters for model initialization by calling "read_parameter_base" to read parameters from neighbourhood
    configuration file, calling "read_parameter_measure" to read parameters from measure configuration file, and
    computing area of xx without measure with given parameters.

    Args:
        stat1_inp (string): filename of neighbourhood configuration file
        measure_id (string): id of measure
        neighbourhood_id (string): id of neighbourhood type

    Returns:
        (dictionary): A dictionary of all necessary parameters to initialize a model
    """
    
    path = Path.cwd() / ".." / "input"
    cf = toml.load(str(path) + "\\" + stat1_inp, _dict=dict)
    # Edit the parameters in the catchment configuration accordingly to the neighbourhood type
    neighbourhood_pars = pd.read_csv('../input/Parameters neighbourhoods.csv')
    idx_neighbourhood = np.where(neighbourhood_pars['id_type']==neighbourhood_id)[0][0]
    for key in neighbourhood_pars:
        cf[key] = neighbourhood_pars[key][idx_neighbourhood]
        
    parameter_base = read_parameter_base_dic(cf)
    parameter_measure = read_parameter_measure_csv(measure_id, parameter_base, apply_measure)
    
    d = dict(pr_no_meas_area=parameter_base["tot_pr_area"] - parameter_measure["pr_meas_area"],
             cp_no_meas_area=parameter_base["tot_cp_area"] - parameter_measure["cp_meas_area"],
             op_no_meas_area=parameter_base["tot_op_area"] - parameter_measure["op_meas_area"],
             up_no_meas_area=parameter_base["tot_up_area"] - parameter_measure["up_meas_area"],
             uz_no_meas_area=parameter_base["tot_uz_area"] - parameter_measure["uz_meas_area"],
             gw_no_meas_area=parameter_base["tot_gw_area"] - parameter_measure["gw_meas_area"],
             swds_no_meas_area=parameter_base["tot_swds_area"] - parameter_measure["swds_meas_area"],
             mss_no_meas_area=parameter_base["tot_mss_area"] - parameter_measure["mss_meas_area"],
             ow_no_meas_area=parameter_base["tot_ow_area"] - parameter_measure["ow_meas_area"],
             )
    rv = {**parameter_base, **parameter_measure, **d}
    # print(rv)
    return rv

@timer

def run_measures(dyn_inp, stat1_inp, measure_id, neighbourhood_id, dyn_out, base_run, varkey, vararrlist1, correspvarkey=None, vararrlist2=None,
                      baseline_variable="r_op_swds", variable_to_save="q_meas_swds"):
    """
    for one type of measure, run a batch of simulations with different values for one (or two) parameter(s)

    Args:
    dyn_inp (string): the filename of the inputdata of precipitation and evaporation
    stat1_inp (string): the filename of the static form of general parameters
    stat2_inp (string): the filename of the static form of measure parameters
    dyn_out (string): the filename of the output file of solutions
    varkey (float): the key parameter to be updated
    vararr (float): values to update varkey

    Usage:
    use in the cmd: python -m urbanwb.main batch_run_measure timeseries.csv stat1.ini stat2.ini results.csv storcap_btm_meas [20,30,40]
    """
    
    inputdata = read_inputdata(dyn_inp)
    dict_param = read_parameters_csv(stat1_inp, measure_id, neighbourhood_id)

    outdir = Path("pysol")
    outdir.mkdir(parents=True, exist_ok=True)

    date = inputdata["date"]

    nameofmeasure = dict_param["title"]
    msg_nameofmeasure = f"Currently running Neighbourhood {str(neighbourhood_id)} - {nameofmeasure}"
    print(msg_nameofmeasure)

    database_runoff = []
    database_gw = []
    database_evap = []
    if correspvarkey is not None:
        for a, b in zip(vararrlist1, vararrlist2):
            dict_param[varkey] = a
            dict_param[correspvarkey] = b
            
            rv = running(inputdata, dict_param)
            results = pd.DataFrame(rv[0]) # Model variables results
            wbc_results = rv[1] # Water Balance values: rv[1][0] = entire model, rv[1][1] = measure itself, rv[1][2] = measure inflow area
            
            avg_p_gw = (results['p_op_gw'].sum() * dict_param['op_no_meas_area'] + results['q_meas_gw'].sum() * dict_param['tot_meas_area'] + results['p_uz_gw'].sum() * dict_param['tot_uz_area']) / dict_param['tot_gw_area']
            
            # Obtain the values of runoff, evaporation and gw recharge of the measure
            database_runoff.append(results[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"])
            database_gw.append(avg_p_gw)
            database_evap.append(wbc_results[0]['evap'])
    else:
        for a in vararrlist1:
            dict_param[varkey] = a
            
            rv = running(inputdata, dict_param)
            results = pd.DataFrame(rv[0]) # Model variables results
            wbc_results = rv[1] # Water Balance values: rv[1][0] = entire model, rv[1][1] = measure itself, rv[1][2] = measure inflow area
            
            avg_p_gw = (results['p_op_gw'].sum() * dict_param['op_no_meas_area'] + results['q_meas_gw'].sum() * dict_param['tot_meas_area'] + results['p_uz_gw'].sum() * dict_param['tot_uz_area']) / dict_param['tot_gw_area']
            
            # Obtain the values of runoff, evaporation and gw recharge of the measure
            runoff = results[variable_to_save]*dict_param["tot_meas_area"]/dict_param["tot_meas_inflow_area"]
            database_runoff.append(runoff)
            database_gw.append(avg_p_gw)
            database_evap.append(wbc_results[0]['evap'])
            
    # Dataframe: runoff  
    df_runoff = pd.DataFrame(database_runoff, index=[v for v in vararrlist1])
    df_runoff = df_runoff.T
    df_runoff.insert(0, "Date", date)
    df_runoff.insert(1, "P_atm", inputdata["P_atm"])
    
    # Dataframe: groundwater recharge
    df_gw = pd.DataFrame(database_gw, index=[v for v in vararrlist1])
    df_gw = df_gw.T
    
    # Dataframe: evaporation
    df_evap = pd.DataFrame(database_evap, index=[v for v in vararrlist1])
    df_evap = df_evap.T

    results_base = pd.DataFrame(base_run[0]) # Model variables results
    wbc_results_base = base_run[1] # Water Balance values for the entire model
    
    # Obtain the values of runoff, evaporation and gw recharge of the baseline
    baseline_runoff = results_base[baseline_variable]
    baseline_gw = results_base['sum_p_gw'].sum()
    baseline_evap = wbc_results_base[0]['evap']
    
    df_runoff.insert(2, "Baseline", baseline_runoff)
    df_gw.insert(0, "Baseline", baseline_gw)
    df_evap.insert(0, "Baseline", baseline_evap)

    return df_runoff, df_gw, df_evap

def read_parameters_exception(stat1_inp, measure_title, neighbourhood_id, apply_measure):
    """
    reads parameters for model initialization by calling "read_parameter_base" to read parameters from neighbourhood
    configuration file, calling "read_parameter_measure" to read parameters from measure configuration file, and
    computing area of xx without measure with given parameters.

    Args:
        stat1_inp (string): filename of neighbourhood configuration file
        stat2_inp (string): filename of measure configuration file

    Returns:
        (dictionary): A dictionary of all necessary parameters to initialize a model
    """
    path = Path.cwd() / ".." / "input"
    cf = toml.load(str(path) + "\\" + stat1_inp, _dict=dict)
    # Edit the parameters in the catchment configuration accordingly to the neighbourhood type
    neighbourhood_pars = pd.read_csv('../input/Parameters neighbourhoods.csv')
    idx_neighbourhood = np.where(neighbourhood_pars['id_type']==neighbourhood_id)[0][0]
    for key in neighbourhood_pars:
        cf[key] = neighbourhood_pars[key][idx_neighbourhood]
    
    measures_exception = pd.read_excel('../input/Parameters measures exception.xlsx', sheet_name=None)
    for key in measures_exception[measure_title]:
        if key == 'title':
            pass
        else:
            if key in cf:
                cf[key] = measures_exception[measure_title][key][0]
            elif key == 'change_op_to_up':
                if measures_exception[measure_title][key][0] == True:
                    cf['up_frac'] += cf['op_frac']
                    cf['op_frac'] = 0
                else:
                    pass
            elif key == 'extra_ow_height':
                cf['storcap_ow'] += measures_exception[measure_title][key][0]
            elif key == 'extra_ow_frac':
                cf['up_frac'] -= measures_exception[measure_title][key][0]
                cf['ow_frac'] += measures_exception[measure_title][key][0]
    
    parameter_base = read_parameter_base_dic(cf)
    parameter_measure = read_parameter_measure_csv(measure_title, parameter_base, apply_measure)
    parameter_measure['title'] = measures_exception[measure_title]['title'][0]
        
    d = dict(pr_no_meas_area=parameter_base["tot_pr_area"] - parameter_measure["pr_meas_area"],
             cp_no_meas_area=parameter_base["tot_cp_area"] - parameter_measure["cp_meas_area"],
             op_no_meas_area=parameter_base["tot_op_area"] - parameter_measure["op_meas_area"],
             up_no_meas_area=parameter_base["tot_up_area"] - parameter_measure["up_meas_area"],
             uz_no_meas_area=parameter_base["tot_uz_area"] - parameter_measure["uz_meas_area"],
             gw_no_meas_area=parameter_base["tot_gw_area"] - parameter_measure["gw_meas_area"],
             swds_no_meas_area=parameter_base["tot_swds_area"] - parameter_measure["swds_meas_area"],
             mss_no_meas_area=parameter_base["tot_mss_area"] - parameter_measure["mss_meas_area"],
             ow_no_meas_area=parameter_base["tot_ow_area"] - parameter_measure["ow_meas_area"],
             )
    rv = {**parameter_base, **parameter_measure, **d}
    # print(rv)
    return rv

# This function implements the measure by changing the catchment properties instead of using measure parameters as input
def run_measures_exception(dyn_inp, stat1_inp, measure_title, neighbourhood_id, base_run, baseline_variable, variable_to_save):
    inputdata = read_inputdata(dyn_inp)
    dict_param = read_parameters_exception(stat1_inp, measure_title, neighbourhood_id, apply_measure=False) # Apply measure is False here, as we change the catchment properties rather than implement an extra measure element
    
    date = inputdata["date"]
    nameofmeasure = dict_param["title"]
    msg_nameofmeasure = f"Currently running Neighbourhood {str(neighbourhood_id)} - {nameofmeasure}"
    print(msg_nameofmeasure)
    
    # Run the model with the new catchment properties
    rv = running(inputdata, dict_param)
    results = rv[0]
    wbc_results = rv[1]
   
    # Obtain runoff. In case of an urban forest, the open pavement is changed to unpaved. In order to compare the runoff values,
    # the added runoff to the unpaved needs to be calculated. This is then compared to the open paved runoff
    runoff = results[variable_to_save]
    
    # Dataframe: runoff
    df_runoff = pd.DataFrame(runoff)
    df_runoff.insert(0, "Date", date)
    df_runoff.insert(1, "P_atm", inputdata["P_atm"])
    
    # Dataframe: groundwater recharge
    gw = rv[0]['sum_p_gw'].sum()
    df_gw = pd.DataFrame([gw], columns=['alt'])
    
    # Dataframe: evaporation
    evap = wbc_results[0]['evap']
    df_evap = pd.DataFrame([evap], columns=['alt'])
    
    # Baseline results for comparison
    results_base = pd.DataFrame(base_run[0]) # Model variables results
    wbc_results_base = base_run[1] # Water Balance values for the entire model
    
    # Obtain the values of runoff, evaporation and gw recharge of the baseline
    baseline_runoff = results_base[baseline_variable]
    baseline_gw = results_base['sum_p_gw'].sum()
    baseline_evap = wbc_results_base[0]['evap']
    
    # Place the baseline values in the dataframes of the effectivity variables
    df_runoff.insert(2, "Baseline", baseline_runoff)
    df_gw.insert(0, "Baseline", baseline_gw)
    df_evap.insert(0, "Baseline", baseline_evap)
    
    return df_runoff, df_gw, df_evap

if __name__ == "__main__":
    fire.Fire()