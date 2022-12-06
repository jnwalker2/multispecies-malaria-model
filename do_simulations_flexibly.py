import H_abm_Mcomixing_pop as sim_codes
import numpy as np
import random
import multiprocessing
import time
import concurrent.futures
import math
from scipy.integrate import solve_ivp
from enum import IntEnum
import json

# import classes I've written
from index_names import Species, Compartments, Mozzie_labels
from disease_model import Disease
from mosquito_model import Mozzies
from human_agents import Agent
from model_parameters import model_parameters

start = time.time()  # for timing the code
fig_dir = "../../figures/code_generated/"
outfilename = "./stored/all_things"
tangled_filename = "./stored/entangled_treatment"

it_dict = {'flag_entangled_treatment': 1}
it_dict["flag_triggering_by_pf"] = 1
it_dict["zf"] = 3.5

import gc

########################### for user to change #################################
prov_list_epidemics = ['Example_Province'] # names need to exist in local_params.json

p_mask = 0.50
mda1 = [270.0]  # lower bound of when MDA occurs
mda2 = [270+30] # upper bound of when MDA occurs (i.e. 30 days of MDA)
c_vec = [[0.3, 0.3]] # coverage scenarios
pP_vec = [0.217]
pG_vec = [[0.0066, 0.000006]]
pN_vec = [[1.0, 0.843]] # treatment scenarios
mda_coverage = [0.5]
pN_mda_vec = [[[1.0, 0.843]]]
MDA_vec = [False]
it_dict["eta"] = [0, 0]
################################################################################

FSAT_vec = [False]
it_dict["etaFSAT"] = [0, 0]
it_dict["etaMDA"] = [0, 0]

in_parallel = True

if __name__ == '__main__':
    for prov_name in prov_list_epidemics:
        calibrated_params, ics = sim_codes.use_calibrated_params(prov=prov_name)
        params = model_parameters(**calibrated_params)

        for iterate1 in range(len(pN_vec)):
            # treatment scenarios

            mask_prob = p_mask * pN_vec[iterate1][0] + (1 - p_mask) * pN_vec[iterate1][1]
            it_dict["pN"] = pN_vec[iterate1]
            it_dict["mask_prob"] = mask_prob
            it_dict["pP"] = pP_vec[iterate1]
            it_dict["pG"] = pG_vec[iterate1]

            for iterate2 in range(len(c_vec)):
                it_dict["etaFSAT"] = [0, 0]
                it_dict["etaMDA"] = [0, 0]
                # coverage scenarios
                it_dict["scenario"]=iterate2
                it_dict["mda_t1"] = mda1[iterate2]
                it_dict["mda_t2"] = mda2[iterate2]

                if MDA_vec[iterate2] == True:
                    it_dict["MDA"] = True
                    it_dict["etaMDA"] = [-math.log(1 - mda_coverage[iterate2]) / (mda2[iterate2] - mda1[iterate2]), -math.log(1 - mda_coverage[iterate2]) / (mda2[iterate2] - mda1[iterate2])]
                else:
                    it_dict["MDA"] = False
                    it_dict["etaMDA"] = [0, 0]

                if FSAT_vec[iterate2] == True:
                    it_dict["FSAT"]=True
                    it_dict["FSAT_period"] = 7
                    it_dict["FSAT_exp_find"] = 10
                    it_dict["localisation"] = 0.4
                else:
                    it_dict["FSAT"] = False
                    it_dict["etaFSAT"] = [0, 0]
                    it_dict["FSAT_period"] = 7
                    it_dict["FSAT_exp_find"] = 0
                    it_dict["localisation"] = 0.4

                it_dict["pN_mda"] = pN_mda_vec[iterate1][iterate2]
                it_dict["mask_prob_mda"] = p_mask * pN_mda_vec[iterate1][iterate2][0] + (1 - p_mask) * pN_mda_vec[iterate1][iterate2][1]
                it_dict["c"] = [c_vec[iterate2][0], c_vec[iterate2][1]]
                sim_codes.do_iterate(params, it_dict, prov_name, in_parallel)
                gc.collect()
