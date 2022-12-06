#!/usr/bin/env python3
# getting a model running with human Agents (SIR) but population of Mozzies (SEI)
import concurrent.futures
import numpy as np
import random
import multiprocessing
import time
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

class Run_Simulations(object):

    def __init__(self, prov_name, **kwargs):
        calibrated_params, self.ics = use_calibrated_params(prov=prov_name)

        self.prov_name = prov_name

        calibrated_params.update(**kwargs)
        self.params = model_parameters(**calibrated_params)
        self.infectious_compartment_list = [Compartments.I, Compartments.A, Compartments.T, Compartments.G]

    def count_agents(self, people, compartment):
        num_pfs = 0
        num_pvs = 0
        for x in range(self.params.human_population):  # NB this includes dead agents, but that's not really a problem
            if people[x].state[Species.falciparum].current == compartment:
                num_pfs += 1
            if people[x].state[Species.vivax].current == compartment:
                num_pvs += 1

        return num_pfs, num_pvs

    def process_death(self, diseases, person):
        """
        Need access to all pathogens in Disease to update all requirement variables.

        I'm assuming the appropriate death.time has been updated where death was determined to occur, given it is specific to pathogen and compartment died from

        Can't update the total human population here or could update partway through this time step

        :param diseases: array of all pathogens being considered, instantiations of Disease
        :param person: the human agent who has died
        :return:
        """
        # update the record for the agent for each pathogen
        for idx in range(self.params.number_pathogens):
            diseases[idx].pop_counts[person.state[idx].current] -= 1  # decrement count
            person.state[idx].current = Compartments.dead
            person.state[idx].next = Compartments.dead
            person.state[idx].time = self.params.time_end + 2  # i.e. never

    def process_treatment_entanglement(self, diseases, person, current_time, event_rates,params):
        """
        If treatment entanglement is on, need to update the status of the other species
        :param diseases:
        :param person:
        :return:
        """
        assert self.params.flag_entangled_treatment == 1

        # identify the species to update
        if person.state[Species.falciparum].current == Compartments.T and person.state[Species.vivax].current != Compartments.T:
            sp = Species.vivax
        else:
            sp = Species.falciparum

        diseases[sp].pop_counts[person.state[sp].current] -= 1  # decrement current compartment count
        diseases[sp].pop_counts[Compartments.T] += 1 # increment treatment compartment count
        person.state[sp].current = Compartments.T  # change to Treatment status
        diseases[sp].transition_table(person=person, current_time=current_time, event_rates=event_rates[sp], params=params)  # determine next compartment

    def process_treatment_entanglement2(self, diseases, person, current_time, event_rates, params):
        """
        If treatment entanglement is on, need to update the status of the other species with radical cure
        :param diseases:
        :param person:
        :return:
        """
        assert self.params.flag_entangled_treatment == 1

        # identify the species to update
        if person.state[Species.falciparum].current == Compartments.G and person.state[Species.vivax].current != Compartments.G:
            sp = Species.vivax
        else:
            sp = Species.falciparum

        diseases[sp].pop_counts[person.state[sp].current] -= 1  # decrement current compartment count
        diseases[sp].pop_counts[Compartments.G] += 1  # increment treatment compartment count
        person.state[sp].current = Compartments.G  # change to Treatment status
        diseases[sp].transition_table(person=person, current_time=current_time, event_rates=event_rates[sp], params=params)  # determine next compartment

    def mixed_infection_event(self, diseases, person, current_time, event_rates,params):
        """
        If treatment entanglement is on, need to update the status of the other species
        :param diseases:
        :param person:
        :return:
        """
        assert self.params.flag_entangled_treatment == 1

        # identify the species to update
        if person.state[Species.falciparum].current == Compartments.T and person.state[Species.vivax].current != Compartments.T:
            sp = Species.vivax
        else:
            sp = Species.falciparum

        diseases[sp].pop_counts[person.state[sp].current] -= 1  # decrement current compartment count
        diseases[sp].pop_counts[Compartments.T] += 1 # increment treatment compartment count
        person.state[sp].current = Compartments.T  # change to Treatment status
        diseases[sp].transition_table(person=person, current_time=current_time, event_rates=event_rates[sp], params=params)  # determine next compartment

    def initialise_agent_compartments(self, rates, diseases, anophs, humans, human_initial_counts):
        """

        :param rates: event_rates so know prob and time 'til next event
        :param diseases: need transition table to calculate next transitions
        :param anophs: mosquito population, needed for the rates
        :param humans: array of human agents
        :param human_initial_counts: assuming this is a full array of counts of falciparum compartments by vivax compartments
        :return: updated agent population with distribution as per human_initial_counts
        """
        comps = [Compartments.S, Compartments.I, Compartments.A, Compartments.R, Compartments.L, Compartments.T, Compartments.G]
        # calculate things needed for all compartments
        pf_starting_event_rates = rates[Species.falciparum](params=self.params, mozzie=anophs, time=0)
        pv_starting_event_rates = rates[Species.vivax](params=self.params, mozzie=anophs, time=0)
        agent_counter = 0  # don't want to overwrite existing allocation
        for pf_cmp in comps:  # falciparum compartment
            for pv_cmp in comps:  # vivax compartment
                if pf_cmp == Compartments.S and pv_cmp == Compartments.S:  # default to here
                    agent_counter += human_initial_counts[pf_cmp][pv_cmp]
                    continue

                for idx in range(human_initial_counts[pf_cmp][pv_cmp]):
                    # update falciparum status
                    humans[agent_counter].state[Species.falciparum].current = pf_cmp
                    diseases[Species.falciparum].transition_table(person=humans[agent_counter], current_time=self.params.time_start,
                                                              event_rates=pf_starting_event_rates, params=self.params)
                    # update vivax status
                    humans[agent_counter].state[Species.vivax].current = pv_cmp
                    diseases[Species.vivax].transition_table(person=humans[agent_counter], current_time=self.params.time_start,
                                                         event_rates=pv_starting_event_rates,
                                                         params=self.params)
                    agent_counter += 1

        assert agent_counter == self.params.human_population, "not all agents assigned since total population = " + str(self.params.human_population) + ", but agents assigned = " + str(agent_counter)
        pf_sums = np.sum(human_initial_counts, axis=1)
        pv_sums = np.sum(human_initial_counts, axis=0)
        for cmp in comps:
            agents_in_pf, agents_in_pv = self.count_agents(people=humans, compartment=cmp)
            assert agents_in_pf == pf_sums[cmp], str(agents_in_pf) + ", " + str(pf_sums[cmp])
            assert agents_in_pv == pv_sums[cmp], str(agents_in_pv) + ", " + str(pv_sums[cmp])

        return humans

    def disperse_ics(self, initial_human_population_counts):
        """
        Takes the initial conditions from the calibration and proportionally disperses them to the full matrix of falciparum by vivax compartments
        :param initial_human_population_counts:
        :return:
        """
        compartments = [Compartments.S, Compartments.I, Compartments.A, Compartments.R, Compartments.L, Compartments.T, Compartments.G]
        num_comps = len(compartments)

        matrix_human_pops = [[0 for idx in range(num_comps)] for idx in range(num_comps)]

        assignT = (initial_human_population_counts[Species.falciparum][Compartments.T] + initial_human_population_counts[Species.vivax][Compartments.T]) / 2
        for pf_comp in compartments:
            for pv_comp in compartments:
                if self.params.flag_entangled_treatment == 1 and pf_comp == Compartments.T and pv_comp == Compartments.T:
                    assign = assignT
                elif self.params.flag_entangled_treatment == 1 and (pf_comp == Compartments.T or pv_comp == Compartments.T):
                    assign = 0  # all treatments are entangled
                else:
                    assign = initial_human_population_counts[Species.falciparum][pf_comp] * initial_human_population_counts[Species.vivax][pv_comp] / self.params.human_population

                matrix_human_pops[pf_comp][pv_comp] = int(round(assign))  # divide by two as for deterministic model used to generate these ICs, the population is independently represented for pf and pv, meaning twice the population when allocated this way.

        pop_diff = self.params.human_population - np.sum(np.sum(matrix_human_pops))
        matrix_human_pops[0][0] = max(0, matrix_human_pops[0][0] + pop_diff)

        infection_compartments = [Compartments.I, Compartments.A, Compartments.T, Compartments.G]
        mixed_infection_pops = [[matrix_human_pops[v_idx][f_idx] for f_idx in infection_compartments] for v_idx in infection_compartments]

        pv_pops = list(np.sum(matrix_human_pops, axis=0))
        pf_pops = list(np.sum(matrix_human_pops, axis=1))
        assert pf_pops[Compartments.L] == 0
        human_population_counts = [pf_pops, pv_pops]

        return matrix_human_pops, mixed_infection_pops, human_population_counts

    def calculate_mixed_only(self, human_initial_mixed, diseases):
        v_sums = np.sum(human_initial_mixed, axis=0)
        f_sums = np.sum(human_initial_mixed, axis=1)
        human_initial_inf_comp_x_only = [[diseases[Species.falciparum].pop_counts[Compartments.I] - f_sums[0], diseases[Species.falciparum].pop_counts[Compartments.A] - f_sums[1], diseases[Species.falciparum].pop_counts[Compartments.T] - f_sums[2],  diseases[Species.falciparum].pop_counts[Compartments.G] - f_sums[3]], [diseases[Species.vivax].pop_counts[Compartments.I] - v_sums[0], diseases[Species.vivax].pop_counts[Compartments.A] - v_sums[1], diseases[Species.vivax].pop_counts[Compartments.T] - v_sums[2], diseases[Species.vivax].pop_counts[Compartments.G] - v_sums[3]]]
        assert all(human_initial_inf_comp_x_only) >= 0.0

        return human_initial_inf_comp_x_only

    def run_me(self, num_repeats):
        """
        :param num_repeats: number of repeats, needed for multiprocessing even though not used
        :return: final size, aggregate of inf human Agents (time series), mozzie population inf (time series)
        """
        ### various counters ###################################################
        num_entangled_T = 0
        num_entangled_G = 0
        num_simultaneous_T = 0
        num_simultaneous_G = 0

        entangled_T_pf = 0   # T for pv. I, A for pf.
        entangled_T_pv = 0   # T for pf. I, A, L for pv.
        entangled_G_pf = 0   # G for pv. I, A for pf.
        entangled_G_pv = 0   # G for pf. I, A, L for pv.

        num_sim_diff_treat = 0

        num_deaths = 0
        death_count_matrix = [[0 for i in range(7)] for j in range(7)]

        false_T_deaths = ([], [])
        false_G_deaths = ([], [])
        false_I_deaths = ([], [])
        double_count_deaths = 0
        deaths_owing = 0
        ########################################################################

        print(num_repeats)

        popn=self.params.human_population

        # use calibrated population ics
        calibrated_human_population_counts = [[0 for idx in range(7)] for idx2 in range(2)]
        for sp in [Species.falciparum, Species.vivax]:
            calibrated_human_population_counts[sp] = [int(self.ics[sp][idx]) for idx in range(7)]

        initial_human_counts_full_combo, human_initial_mixed_all, initial_human_counts_single_sp = self.disperse_ics(calibrated_human_population_counts)

        mozzie_initial_inf = [self.ics[Species.falciparum][-1], self.ics[Species.vivax][-1], 0.0]
        initial_mozzie = [0.0, self.ics[Species.falciparum][-2], mozzie_initial_inf[0], self.ics[Species.vivax][-2], mozzie_initial_inf[1], 0.0, 0.0, 0.0, 0]
        initial_mozzie[0] = self.params.mozzie_pop - sum(initial_mozzie)
        assert all(initial_mozzie) >= 0.0

        # preallocating
        # humans
        initial_human_counts_single_sp[Species.falciparum].append(0)
        initial_human_counts_single_sp[Species.falciparum].append(0)
        initial_human_counts_single_sp[Species.vivax].append(0)
        initial_human_counts_single_sp[Species.vivax].append(0)

        human_pop_pf_history = [None] * self.params.time_end  # preallocate memory
        human_pop_pf_history[0] = initial_human_counts_single_sp[Species.falciparum].copy()  # add initial counts

        human_pop_pv_history = [None] * self.params.time_end  # preallocate memory
        human_pop_pv_history[0] = initial_human_counts_single_sp[Species.vivax].copy()  # add initial counts

        human_pop_mixed_inf_history = [None] * self.params.time_end
        human_pop_mixed_inf_history[0] = human_initial_mixed_all

        # mozzies
        mozzie_pop_inf_history = [None, None, None] * self.params.time_end
        mozzie_pop_inf_history[0] = mozzie_initial_inf.copy()
        mozzie_pop_history = [None] * self.params.time_end
        mozzie_pop_history[0] = initial_mozzie.copy()

        # initialise Disease objects
        diseases = (Disease(malaria_species=Species.falciparum, initial_population_counts=initial_human_counts_single_sp[Species.falciparum]), Disease(malaria_species=Species.vivax, initial_population_counts=initial_human_counts_single_sp[Species.vivax]))
        update = (diseases[Species.falciparum].update, diseases[Species.vivax].update)
        rates = (diseases[Species.falciparum].stochastic_sir_event_rates, diseases[Species.vivax].stochastic_sir_event_rates)
        human_initial_inf_comp_x_only = self.calculate_mixed_only(human_initial_mixed_all, diseases)


        # initialise populations
        # mozzies
        anophs = Mozzies(mozzie_initial_inf=mozzie_initial_inf, human_initial_inf_comp_x_only=human_initial_inf_comp_x_only, human_initial_mixed=human_initial_mixed_all)
        anophs_y0 = initial_mozzie
        # humans
        humans = [Agent(transition_time=self.params.time_end+1) for x in range(self.params.human_population)]
        humans = self.initialise_agent_compartments(rates=rates, diseases=diseases, anophs=anophs, humans=humans, human_initial_counts=initial_human_counts_full_combo)

        FSAT_indicator = np.zeros(self.params.time_end + self.params.FSAT_period)

        # iterate through time
        mozzie_model = anophs.model
        for t in range(1, self.params.time_end):  # from 1 as initial conditions at time 0
            if self.params.FSAT == True:
                possible_detections = human_pop_pf_history[t - 1][1] + human_pop_pf_history[t - 1][2] + \
                                      human_pop_pv_history[t - 1][1] + human_pop_pv_history[t - 1][2] - \
                                      human_pop_mixed_inf_history[t - 1][0][0] - human_pop_mixed_inf_history[t - 1][0][
                                          1] - human_pop_mixed_inf_history[t - 1][1][0] - \
                                      human_pop_mixed_inf_history[t - 1][1][1]
                self.params.etaFSAT[0] = self.params.FSAT_exp_find * (FSAT_indicator[t-1]/self.params.FSAT_period) / (human_pop_pf_history[t - 1][1] + human_pop_pf_history[t - 1][2] + human_pop_pv_history[t - 1][1] + human_pop_pv_history[t - 1][2]) * (possible_detections/ self.params.human_population) ** self.params.localisation
                self.params.etaFSAT[1] = self.params.FSAT_exp_find * (FSAT_indicator[t-1] / self.params.FSAT_period) / (human_pop_pf_history[t - 1][1] + human_pop_pf_history[t - 1][2]+human_pop_pv_history[t - 1][1] + human_pop_pv_history[t - 1][2]) * (possible_detections / self.params.human_population) ** self.params.localisation
            current_event_rates = (rates[Species.falciparum](params=self.params, mozzie=anophs, time=t), rates[Species.vivax](params=self.params, mozzie=anophs, time=t))  # based on mozzie numbers previous time step
            # for each human Agent -- below comment just for reference of indexes in mixed_counter
            # [[if_iv=[0][0], if_av=[0][1], if_tv=[0][2], if_gv=[0][3]],
            #  [af_iv=[1][0], af_av=[1][1], af_tv=[1][2], af_gv=[0][3]],
            #  [tf_iv=[2][0], tf_av=[2][1], tf_tv=[2][2], tf_gv=[2][3]],
            #  [gf_iv=[3][0], gf_av=[3][1], gf_tv=[3][2], gf_gv=[3][3]]]
            mixed_counter = [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
            agent_death_counter = 0
            for human in humans:

                prev_pf=human.state[Species.falciparum].current
                prev_pv = human.state[Species.vivax].current

                update[Species.falciparum](person=human, current_time=t, event_rates=current_event_rates[Species.falciparum], params=self.params, event_rates_other_sp=current_event_rates[Species.vivax])
                update[Species.vivax](person=human, current_time=t, event_rates=current_event_rates[Species.vivax], params=self.params)

                # Logic for setting a mixed infection transition
                curr_pf = human.state[Species.falciparum].current
                curr_pv = human.state[Species.vivax].current

                # count simultaneous treatments
                if (prev_pf != Compartments.T and prev_pv != Compartments.T and curr_pf == Compartments.T and curr_pv == Compartments.T):
                    num_simultaneous_T += 1
                if (prev_pf != Compartments.G and prev_pv != Compartments.G and curr_pf == Compartments.G and curr_pv == Compartments.G):
                    num_simultaneous_G += 1

                ### prevent treatment re-entry loop ############################
                treatment_compartments = [Compartments.T, Compartments.G]
                exit_compartments = [Compartments.A, Compartments.R, Compartments.L]

                # consider if a human enters T for one species and G for the other in the same step
                if (curr_pf == Compartments.G and curr_pv == Compartments.T) or (curr_pf == Compartments.T and curr_pv == Compartments.G):
                    if (prev_pf not in treatment_compartments) and (prev_pv not in treatment_compartments):
                        num_sim_diff_treat += 1

                # single species treatment exit
                if prev_pf in treatment_compartments and prev_pv in treatment_compartments:
                    if curr_pf in treatment_compartments and curr_pv in exit_compartments:
                        sp_ = Species.falciparum
                    elif curr_pf in exit_compartments and curr_pv in treatment_compartments:
                        sp_ = Species.vivax
                    else:
                        sp_ = 'non-exit'
                    # force treatment exit
                    if sp_ != 'non-exit':
                        # log overidden deaths
                        if human.state[sp_].next == Compartments.just_died:
                            if human.state[sp_].current == Compartments.T:
                                false_T_deaths[sp_].append(human.state[sp_].time)
                            else: # human.state[sp_].current == Compartments.G
                                false_G_deaths[sp_].append(human.state[sp_].time)

                        # probabilites for transitions
                        if human.state[sp_].current == Compartments.T:
                            prob_treatment_death = self.params.pT[sp_]
                            prob_treatment_failure = self.params.pTfA[sp_]
                            prob_treatment_latency = self.params.pA[sp_]
                        else: # human.state[sp_].current == Compartments.G
                            prob_treatment_death = self.params.pG[sp_]
                            prob_treatment_failure = self.params.pTfP[sp_]
                            prob_treatment_latency = self.params.pP[sp_]
                        # move compartments
                        diseases[sp_].pop_counts[human.state[sp_].current] -= 1
                        if random.random() < prob_treatment_death: # death
                            if human.state[sp_].current == Compartments.T:
                                diseases[sp_].T_deaths.append(t)
                            else: # human.state[sp_].current == Compartments.G
                                diseases[sp_].G_deaths.append(t)
                            human.state[sp_].current = Compartments.just_died
                        else:
                            if random.random() < prob_treatment_failure: # treatment failure
                                human.state[sp_].current = Compartments.A
                            else:
                                if random.random() < prob_treatment_latency: # latency
                                    human.state[sp_].current = Compartments.L
                                    assert sp_ == Species.vivax, "shouldn't be assigning compartment L to falciparum"
                                else: # recovery
                                    human.state[sp_].current = Compartments.R
                        # update counts and compute next transition
                        diseases[sp_].pop_counts[human.state[sp_].current] += 1
                        if human.state[sp_].current == Compartments.just_died:
                            # counters
                            num_deaths += 1
                            death_count_matrix[prev_pf][prev_pv] += 1
                            # necessary for model
                            agent_death_counter += 1
                            self.process_death(diseases=diseases, person=human)
                        else:
                            diseases[sp_].transition_table(person=human, current_time=t, event_rates=current_event_rates[sp_], params=self.params)
                ################################################################
                # Logic for setting a mixed infection transition
                if curr_pf == prev_pf and curr_pv == prev_pv:
                    if (curr_pv == Compartments.S or curr_pv == Compartments.R or curr_pv == Compartments.L) and (curr_pf == Compartments.S or curr_pf == Compartments.R): #they are a candidate for mixed infection
                        #rate of mixed infection
                        if type(self.params.b) == list:
                            lamx = 0.5*(self.params.b[Species.falciparum] + self.params.b[Species.vivax] )* self.params.epsilonM[Species.falciparum] * self.params.epsilonM[Species.vivax] *anophs.mozzie_I_count[Species.mixed] / self.params.human_population
                        else:
                            lamx = self.params.b * self.params.epsilonM[Species.falciparum] * self.params.epsilonM[Species.vivax] *anophs.mozzie_I_count[Species.mixed] / self.params.human_population

                        if random.random() < (1 - math.exp(-lamx)):
                            sp_vec = [Species.falciparum, Species.vivax]
                            for sp in sp_vec:

                                diseases[sp].new_infections.append(t)  # just moved to I or A *now*
                                diseases[sp].pop_counts[curr_pf] -= 1  # decrement count for current state
                                if human.state[sp].current == Compartments.S:
                                    prob_I = self.params.pc[sp]
                                elif human.state[sp].current == Compartments.R:
                                    prob_I = self.params.pR[sp]
                                else:
                                    prob_I = self.params.pL[sp]

                                if random.random() <= prob_I:
                                    human.state[sp].current = Compartments.I
                                    diseases[sp].new_inf_clinical.append(t)  # just moved to I *now*
                                else:
                                    human.state[sp].current = Compartments.A

                                diseases[sp].pop_counts[human.state[sp].current] += 1  # increment new current state
                                diseases[sp].transition_table(person=human, current_time=t, event_rates=current_event_rates[sp], params=self.params)

                # check if agent died, if so update everything
                if human.state[Species.falciparum].current == Compartments.just_died or human.state[Species.vivax].current == Compartments.just_died:
                    # death counters
                    num_deaths += 1
                    death_count_matrix[prev_pf][prev_pv] += 1
                    agent_death_counter += 1
                    if human.state[Species.falciparum].current == Compartments.just_died and human.state[Species.vivax].current == Compartments.just_died:
                        double_count_deaths += 1
                    if human.state[Species.falciparum].next == Compartments.just_died or human.state[Species.vivax].next == Compartments.just_died:
                        double_count_deaths += 1
                    # update agent state
                    self.process_death(diseases=diseases, person=human)

                # entangled treatment logic (sends people with infection to treatment states)
                if self.params.flag_entangled_treatment == 1:
                    if (human.state[Species.falciparum].current == Compartments.T and (human.state[Species.vivax].current == Compartments.I or human.state[Species.vivax].current == Compartments.A or human.state[Species.vivax].current == Compartments.L)) or (human.state[Species.vivax].current == Compartments.T and (human.state[Species.falciparum].current == Compartments.I or human.state[Species.falciparum].current == Compartments.A)):
                        # entangled treatment counters
                        num_entangled_T += 1
                        if (human.state[Species.falciparum].current == Compartments.T and (human.state[Species.vivax].current == Compartments.I or human.state[Species.vivax].current == Compartments.A or human.state[Species.vivax].current == Compartments.L)):
                            # T entanglement of pv
                            entangled_T_pv += 1
                            # log overridden deaths
                            if human.state[Species.vivax].next == Compartments.just_died:
                                assert human.state[Species.vivax].current == Compartments.I, 'death from A or L'
                                false_I_deaths[Species.vivax].append(human.state[Species.vivax].time)
                        else:
                            # T entanglement of pf
                            entangled_T_pf += 1
                            # log overridden deaths
                            if human.state[Species.falciparum].next == Compartments.just_died:
                                assert human.state[Species.falciparum].current == Compartments.I, 'death from A or L'
                                false_I_deaths[Species.falciparum].append(human.state[Species.falciparum].time)
                        # update agent state
                        self.process_treatment_entanglement(diseases=diseases, person=human, current_time=t, event_rates=current_event_rates, params=self.params)
                    elif (human.state[Species.falciparum].current == Compartments.G and (human.state[Species.vivax].current == Compartments.I or human.state[Species.vivax].current == Compartments.A or human.state[Species.vivax].current == Compartments.L)) or (human.state[Species.vivax].current == Compartments.G and (human.state[Species.falciparum].current == Compartments.I or human.state[Species.falciparum].current == Compartments.A)):
                        # entangled treatment counters
                        num_entangled_G += 1
                        if (human.state[Species.falciparum].current == Compartments.G and (human.state[Species.vivax].current == Compartments.I or human.state[Species.vivax].current == Compartments.A or human.state[Species.vivax].current == Compartments.L)):
                            # G entanglement of pv
                            entangled_G_pv += 1
                            # log overridden deaths
                            if human.state[Species.vivax].next == Compartments.just_died:
                                assert human.state[Species.vivax].current == Compartments.I, 'death from A or L'
                                false_I_deaths[Species.vivax].append(human.state[Species.vivax].time)
                        else:
                            # G entanglement of pf
                            entangled_G_pf += 1
                            # log overidden deaths
                            if human.state[Species.falciparum].next == Compartments.just_died:
                                assert human.state[Species.falciparum].current == Compartments.I, 'death from A or L'
                                false_I_deaths[Species.falciparum].append(human.state[Species.falciparum].time)
                        # update agent state
                        self.process_treatment_entanglement2(diseases=diseases, person=human, current_time=t, event_rates=current_event_rates, params=self.params)

                # need to count mixed infections for mozzie population update
                current_f_state = human.state[Species.falciparum].current
                current_v_state = human.state[Species.vivax].current
                if current_f_state in self.infectious_compartment_list and current_v_state in self.infectious_compartment_list:
                    mixed_counter[self.infectious_compartment_list.index(current_f_state)][self.infectious_compartment_list.index(current_v_state)] += 1
                if self.params.FSAT == True:
                    if (current_f_state == Compartments.T and prev_pf != current_f_state) or (current_f_state == Compartments.G and prev_pf != current_f_state) or (current_v_state == Compartments.T and prev_pv != current_v_state) or (current_v_state == Compartments.G and prev_pv != current_v_state):
                        FSAT_indicator[t:(t+self.params.FSAT_period)] = FSAT_indicator[t:(t+self.params.FSAT_period)]+1

	        # update time-based record
            assert all(diseases[Species.falciparum].pop_counts) >= 0
            assert all(diseases[Species.vivax].pop_counts) >= 0
            human_pop_pf_history[t] = diseases[Species.falciparum].pop_counts.copy()
            human_pop_pv_history[t] = diseases[Species.vivax].pop_counts.copy()
            human_pop_mixed_inf_history[t] = mixed_counter #sum([sum(mixed_counter[idx]) for idx in range(4)])

            # now update the Mozzies
            # solve ODEs using previous timestep human counts
            mozzie_sol = solve_ivp(fun=lambda t, y: mozzie_model(t, y, params=self.params), t_span=[t-1, t], y0=anophs_y0)
            assert (mozzie_sol.y >= 0).all(), ["mozzie population went negative at t=" + str(t)]

            # update Mozzie class record of counts using this timestep update
            anophs.human_comp_x_only = self.calculate_mixed_only(mixed_counter, diseases)
            anophs.human_compx_compy = mixed_counter
            anophs_y0 = [mozzie_sol.y[idx][-1] for idx in range(9)]  # initial condition for next run is final value from this one
            anophs.mozzie_I_count = [(anophs_y0[Mozzie_labels.Yf] + anophs_y0[Mozzie_labels.Z_yf_wv]), (anophs_y0[Mozzie_labels.Yv] + anophs_y0[Mozzie_labels.Z_wf_yv]), anophs_y0[Mozzie_labels.Yfv]]
            mozzie_pop_inf_history[t] = anophs.mozzie_I_count
            mozzie_pop_history[t] = anophs_y0  # this is the current solution

            # now update human population count for next timestep
            self.params.human_population -= agent_death_counter

        # death counters
        for human in humans:
            if human.state[Species.falciparum].next == Compartments.just_died:
                deaths_owing += 1
            if human.state[Species.vivax].next == Compartments.just_died:
                deaths_owing += 1

        # outputs
        pf_outcomes = (diseases[Species.falciparum].new_infections, diseases[Species.falciparum].new_inf_clinical, diseases[Species.falciparum].relapses, diseases[Species.falciparum].new_T, diseases[Species.falciparum].new_G, diseases[Species.falciparum].T_deaths, diseases[Species.falciparum].G_deaths, diseases[Species.falciparum].I_deaths)

        pv_outcomes = (diseases[Species.vivax].new_infections, diseases[Species.vivax].new_inf_clinical, diseases[Species.vivax].relapses, diseases[Species.vivax].new_T, diseases[Species.vivax].new_G, diseases[Species.vivax].T_deaths, diseases[Species.vivax].G_deaths, diseases[Species.vivax].I_deaths)

        num_TGD = (num_entangled_T, num_entangled_G, entangled_T_pf, entangled_T_pv, entangled_G_pf, entangled_G_pv, num_simultaneous_T, num_simultaneous_G, num_sim_diff_treat, num_deaths, double_count_deaths, deaths_owing, false_T_deaths, false_G_deaths, false_I_deaths, death_count_matrix)

        return human_pop_pf_history, human_pop_pv_history, human_pop_mixed_inf_history, mozzie_pop_inf_history, mozzie_pop_history, pf_outcomes, pv_outcomes, humans, anophs, num_TGD ### total T, G, death: added [9] ###

# read parameters from calibrated values
def use_calibrated_params(prov):

    params = dict()

    # update the default parameter values using parameter values stored in `./local_params.json`, after `parameter-play.py` processes the values in `./stored/model_calibration_params.json`, which were generated from `calibrated_to_cambodia_data.py`,
    with open('stored/local_params.json') as json_file:
        json_data = json.load(json_file)

    for keys in json_data[prov]:
        params[keys] = json_data[prov][keys]

    ics = params['ics']
    del params['ics']

    return params, ics

def convert(o):
    """ from: https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python"""
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def do_iterate(params, it_dict, prov_name, in_parallel):
    print('beginning a stochastic run')

    max_values_f = []
    max_values_v = []
    av_values_f = []
    av_values_v = []
    min_values_f = []
    min_values_v = []

    human_mixed_infectious =  []
    human_pf = []
    human_pv = []
    mozzie_pf_infectious = []
    mozzie_pv_infectious = []
    mozzie_mixed_infectious = []
    mozzie_all = []
    pf_outcomes = []
    pv_outcomes = []
    human_agents = []
    mozzie_info = []
    num_TGD = []

    # parallelised
    if in_parallel==False:
        start = time.time()
        for f in [Run_Simulations(prov_name, **it_dict).run_me(x) for x in range(params.number_repeats)]:

            human_pf.append(f[0])
            human_pv.append(f[1])
            human_mixed_infectious.append(f[2])
            mozzie_pf_infectious.append([f[3][idx][Species.falciparum] + f[3][idx][Species.mixed] for idx in range(params.time_end)])
            mozzie_pv_infectious.append([f[3][idx][Species.vivax] + f[3][idx][Species.mixed] for idx in range(params.time_end)])
            mozzie_mixed_infectious.append([f[3][idx][Species.mixed] for idx in range(params.time_end)])
            mozzie_all.append(f[4])
            pf_outcomes.append(f[5])
            pv_outcomes.append(f[6])
            human_agents.append(f[7])
            mozzie_info.append(f[8])
            num_TGD.append(f[9])
    else:
        start = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:

            results_test = executor.map(Run_Simulations(prov_name, **it_dict).run_me, range(params.number_repeats))

            for f in results_test:
                human_pf.append(f[0])
                human_pv.append(f[1])
                human_mixed_infectious.append(f[2])
                mozzie_pf_infectious.append([f[3][idx][Species.falciparum]+f[3][idx][Species.mixed] for idx in range(params.time_end)])
                mozzie_pv_infectious.append([f[3][idx][Species.vivax]+f[3][idx][Species.mixed] for idx in range(params.time_end)])
                mozzie_mixed_infectious.append([f[3][idx][Species.mixed] for idx in range(params.time_end)])
                mozzie_all.append(f[4])
                pf_outcomes.append(f[5])
                pv_outcomes.append(f[6])
                human_agents.append(f[7])
                mozzie_info.append(f[8])
                num_TGD.append(f[9])

    end = time.time()  # for timing the code
    print('finished in: ' + str(end - start) + 'seconds')

    print('Finished stochastic runs')

    outfilename = "./stored/all_things"
    tangled_filename = "./stored/entangled_treatment"

    # saving results to file
    results = {'human_pop_pf_history': human_pf, 'human_pop_pv_history': human_pv, 'human_pop_mixed_inf_history': human_mixed_infectious, 'mozzie_pf_infectious': mozzie_pf_infectious, 'mozzie_pv_infectious': mozzie_pv_infectious, 'mozzie_mixed_infectious': mozzie_mixed_infectious, 'mozzie_pop_history': mozzie_all, 'pf_outcomes': pf_outcomes, 'pv_outcomes': pv_outcomes, 'num_TGD': num_TGD} #, 'agent_info': human_agents, 'mozzie_info': mozzie_info, 'num_TGD': num_TGD}

    print(it_dict)
    if it_dict['flag_entangled_treatment'] == 1:
        treatment_entangled = True
    else:
        treatment_entangled = False

    if treatment_entangled == True:
        with open(tangled_filename + prov_name + '_all_pN' + str(it_dict['pN'][0]) + "_" + str(it_dict['pN'][1]) + "_scenario" + str(it_dict["scenario"]) + ".json", 'w') as outfile:
            json.dump(results, outfile, indent=4, default=convert)
    else:
        with open(outfilename + prov_name + '_all_pN' + str(it_dict['pN'][0]) + "_" + str(it_dict['pN'][1]) + "_c" + str(it_dict['c'][0]) + ".json", 'w') as outfile:
            json.dump(results, outfile, indent=4, default=convert)

    # store interesting values
    max_values_f.append([list(x) for x in np.amax(human_pf, axis=0)])
    max_values_v.append([list(x) for x in np.amax(human_pv, axis=0)])
    av_values_f.append([list(x) for x in np.mean(human_pf, axis=0)])
    av_values_v.append([list(x) for x in np.mean(human_pv, axis=0)])
    min_values_f.append([list(x) for x in np.amin(human_pf, axis=0)])
    min_values_v.append([list(x) for x in np.amin(human_pv, axis=0)])

    iterate_results = {'max_pf': max_values_f, 'max_pv': max_values_v, 'average_pf': av_values_f, 'average_pv': av_values_v, 'min_pf': min_values_f, 'min_pv': min_values_v, 'iterate': it_dict}

    if treatment_entangled == True:
        with open(tangled_filename + prov_name + '_pN' + str(it_dict['pN'][0]) + "_" + str(it_dict['pN'][1]) + "_c" + str(it_dict['c'][0]) + ".json", 'w') as outfile:
            json.dump(iterate_results, outfile, indent=4, default=convert)
    else:
        with open(outfilename + prov_name + '_pN' + str(it_dict['pN'][0]) + "_" + str(it_dict['pN'][1]) + "_c" + str(it_dict['c'][0]) + ".json", 'w') as outfile:
            json.dump(iterate_results, outfile, indent=4, default=convert)
