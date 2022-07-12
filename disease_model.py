from index_names import Compartments, Transitions, Species
import random
import math

class Disease(object):
    """
    Keep track of human population level aggregates and logic for updating an Agent
    """
    __slots__ = ('pop_counts', 'total_humans', 'species', 'T_deaths', 'G_deaths', 'I_deaths', 'new_infections',
                 'relapses', 'new_inf_clinical', 'new_T', 'new_G')

    def __init__(self, malaria_species, initial_population_counts):
        """
        Each of the possible Disease states have been pre-allocated as slots, population level aggegrates recorded for use here
        :param malaria_species: which malaria species this instantiation is for
        :param initial_population_counts: initial human population in each compartment
        """
        self.pop_counts = initial_population_counts  # last one is for `just_died`
        self.species = malaria_species

        # things tracking for later analysis. i.e. `outputs of interest`
        self.T_deaths = []
        self.G_deaths = []
        self.I_deaths = []
        self.new_infections = []
        self.new_inf_clinical = []
        self.relapses = []
        self.new_T = []
        self.new_G = []

    def calculate_time(self, current_time, rate):
        return current_time + int(max(1, round(random.expovariate(rate), ndigits=0)))

    def transition_table(self, person, current_time, event_rates, params):
        """
        Place to specify transition table: Note the fact that a transition is/has just occurred was determined in self.update() and infection events are determined in self.infect_me()
        :param person: current agent
        :param current_time: present simulation time step (integer)
        :param event_rates: transition rates pre-calculated from stochastic_sir_event_rates() using population level stats from previous time step
        :param params: the model parameters
        :return: nothing explicit, but next agent compartment and.time updated
        """
        current_status = person.state[self.species].current  # just transitioned to this
        # determine what next transition is, and when (excepting infection)
        if current_status == Compartments.S:
            _next = Compartments.S
            _time = params.time_end + 1  # stay susceptible unless infected

        elif current_status == Compartments.I:
            # I->death/A or I->T/G happens first
            time_next = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.I_next])
            time_treat = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.I_treat])

            # MDA should affect the I compartment too
            if event_rates[Transitions.A_treat] == 0:  # avoiding a divide by zero error
                time_mda = 3 * params.time_end  # i.e. never, but making sure likely to be bigger than time_recover
            else:
                time_mda = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.A_treat])

            # if transition before mda or treatment progress
            if time_next < time_treat and time_next<time_mda:
                _time = time_next
                if random.random() < params.pI[self.species]:  # malaria death
                    self.I_deaths.append(_time)
                    _next = Compartments.just_died
                else:
                    _next = Compartments.A
            elif time_treat<time_mda:
                _time = time_treat

                # MASKING LOGIC
                if (person.state[Species.falciparum].current == Compartments.I or person.state[Species.falciparum].current == Compartments.A) and (person.state[Species.vivax].current == Compartments.I or person.state[Species.vivax].current == Compartments.A): #if mixed infection masking is possible
                    if random.random() < params.mask_prob: #treat with prob treat with T given masking may occur
                        _next = Compartments.T
                    else: # if masking doesn't happen treat as usual
                        _next = Compartments.G

                else:
                    if random.random() < params.pN[self.species]:
                        _next = Compartments.T
                    else:
                        _next = Compartments.G
            else:
                _time = time_mda

                # MASKING LOGIC
                if (person.state[Species.falciparum].current == Compartments.I or person.state[Species.falciparum].current == Compartments.A) and (person.state[Species.vivax].current == Compartments.I or person.state[Species.vivax].current == Compartments.A): #if mixed infection masking is possible
                    if random.random() < params.mask_prob_mda: # treat with prob treat with T given masking may occur
                        _next = Compartments.T
                    else: # if masking doesn't happen treat as usual
                        _next = Compartments.G

                else:
                    if random.random() < params.pN_mda[self.species]:
                        _next = Compartments.T
                    else:
                        _next = Compartments.G

        elif current_status == Compartments.A:
            # A-> L or R, or T or G
            time_recover = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.A_recover])
            if event_rates[Transitions.A_treat] == 0:  # avoiding a divide by zero error
                time_mda = 3 * params.time_end  # i.e. never, but making sure likely to be bigger than time_recover
            else:
                time_mda = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.A_treat])

            if time_recover < time_mda:
                _time = time_recover
                if random.random() < params.ph[self.species]:
                    _next = Compartments.L
                    assert self.species == Species.vivax, "falciparum being allocated to L in Disease.transitions_table from A"
                else:
                    _next = Compartments.R
            else:
                # note that this implies that all MDA events are ACT # put in masking
                _time = time_mda
                if random.random() < params.pN_mda[self.species]:
                    _next = Compartments.T
                else:
                    _next = Compartments.G

        elif current_status == Compartments.R:
            # unless infected before this happens, immunity wanes
            _next = Compartments.S
            _time = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.R_S])

        elif current_status == Compartments.L:
            assert self.species == Species.vivax, "falciparum was just assigned to `L`"
            # either relapse (I or A) or hypnozoite death (S) (unless infected in the meantime)
            time_relapse = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.L_relapse])
            time_recover = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.L_S])
            if time_relapse < time_recover:
                _time = time_relapse
                if random.random() < params.pL[self.species]:
                    _next = Compartments.I
                else:
                    _next = Compartments.A
            else:
                _time = time_recover
                _next = Compartments.S

        elif current_status == Compartments.T:
            self.new_T.append(current_time)  # just got to treatment
            _time = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.T_done])
            if random.random() < params.pT[self.species]:  # malaria death
                self.T_deaths.append(person.state[self.species].time)
                _next = Compartments.just_died  # a flag in run_me to update all pathogen compartments and human population size as recoreded in Mozzies
            else:  # 1 - pT
                if random.random() < params.pTfA[self.species]:  # treatment failure
                    _next = Compartments.A
                else:  # 1 - pTfA
                    if random.random() < params.pA[self.species]:  # recover with hypnozoites
                        _next = Compartments.L
                        assert self.species == Species.vivax, "shouldn't be assigning compartment L to falciparum in Disease.transition_table, from T"
                    else:  # 1 - pA :: recover with no hypnozoites
                        _next = Compartments.R

        elif current_status == Compartments.G:
            self.new_G.append(current_time)  # record now getting `G` treatment
            _time = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.G_done])
            if random.random() < params.pG[self.species]:  # malaria death
                self.G_deaths.append(person.state[self.species].time)
                _next = Compartments.just_died  # remove from compartments do anything with
            else:  # 1 - pG
                if random.random() < params.pTfP[self.species]:  # treatment failure
                    _next = Compartments.A
                else:  # 1 - pTfP
                    if random.random() < params.pP[self.species]:  # recover with hypnozoites
                        _next = Compartments.L
                        assert self.species == Species.vivax, "shouldn't be assigning compartment L to falciparum in Disease.transition_table, from compartment G"
                    else:  # 1 - pP :: recover without hypnozoites
                        _next = Compartments.R

        elif current_status == Compartments.just_died:  # person just died
            _next = Compartments.dead
            _time = current_time  # using this to catch if not moved to `dead` and hence agent and disease updated for all pathogens in `run_me`
        else:
            raise ValueError("A compartment that doesn't exist has been assigned and sent to Disease.transmission_table()")

        # make the specified updates
        person.state[self.species].next = _next
        person.state[self.species].time = _time

    def triggering(self, person, current_time, event_rates, params):
        """
        Re-doing transition from current L to elsewhere due to recent falciparum infection, with increased nu_hat = Zf * nu
        :param person:
        :param current_time:
        :param event_rates:
        :param params:
        :return:
        """
        # either relapse (I or A) or hypnozoite death (S) (unless infected in the meantime)
        time_relapse = self.calculate_time(current_time=current_time, rate=params.zf * event_rates[Transitions.L_relapse])
        time_recover = self.calculate_time(current_time=current_time, rate=event_rates[Transitions.L_S])
        if time_relapse < time_recover:
            _time = time_relapse
            if random.random() < params.pL[self.species]:
                _next = Compartments.I
            else:
                _next = Compartments.A
        else:
            _time = time_recover
            _next = Compartments.S

        person.state[Species.vivax].next = _next
        person.state[Species.vivax].time = _time

    def update(self, person, current_time, event_rates, params, event_rates_other_sp = None):
        """
        check if time to next event is now, or infection occurs
        :return: updates state
        """
        assert person.state[self.species].time >= current_time
        current_compartment = person.state[self.species].current

        if person.state[self.species].time == current_time:  # an event has been scheduled for this time
            assert current_compartment != Compartments.dead or person.state[self.species].current != Compartments.just_died
            # decrement population count for current state

            self.pop_counts[current_compartment] -= 1
            # increment population count for next state

            self.pop_counts[person.state[self.species].next] += 1

            if current_compartment == Compartments.L and (person.state[self.species].next == Compartments.I or person.state[self.species].next == Compartments.A):
                self.relapses.append(current_time)  # record relapse event -- here and not in self.update as a new infection may occur in the meantime

            # add triggering logic (triggering occurs after a pf recovery)
            if params.flag_triggering_by_pf and self.species == Species.falciparum and person.state[Species.vivax].current == Compartments.L and person.state[self.species].next == Compartments.R:
                self.triggering(person=person, current_time=current_time, event_rates=event_rates_other_sp, params=params)

            # continue as usual
            person.state[self.species].current = person.state[self.species].next  # transition occurs
            self.transition_table(person=person, current_time=current_time, event_rates=event_rates, params=params)  # identify next transition to occur
        else:   # check for infection event
            if current_compartment == Compartments.S:
                self.infect_me(person=person, rate_infection=event_rates[Transitions.S_inf], prob_I=params.pc[self.species], params=params, current_time=current_time, event_rates=event_rates)
            elif current_compartment == Compartments.R:
                self.infect_me(person=person, rate_infection=event_rates[Transitions.R_inf], prob_I=params.pR[self.species], params=params, current_time=current_time, event_rates=event_rates)
            elif current_compartment == Compartments.L:
                assert self.species == Species.vivax, "falciparum person is in `L`"
                self.infect_me(person=person, rate_infection=event_rates[Transitions.L_inf], prob_I=params.pL[self.species], params=params, current_time=current_time, event_rates=event_rates)

    def infect_me(self, person, rate_infection, prob_I, params, current_time, event_rates):
        if random.random() < (1 - math.exp(-rate_infection)):
            self.new_infections.append(current_time)  # just moved to I or A *now*
            self.pop_counts[person.state[self.species].current] -= 1  # decrement count for current state
            if random.random() <= prob_I:
                person.state[self.species].current = Compartments.I
                self.new_inf_clinical.append(current_time)  # just moved to I *now*
            else:
                person.state[self.species].current = Compartments.A

            self.pop_counts[person.state[self.species].current] += 1  # increment new current state
            self.transition_table(person=person, current_time=current_time, event_rates=event_rates, params=params)  # determine next event and time it occurs

    def stochastic_sir_event_rates(self, params, mozzie, time):
        """
        Calculate the rates events happen for humans in this time step
        Moved to within the disease class so can have species specific parameter values and events
        :param params: model parameters
        :param params: Mozzie class holds the population level counts needed
        :return: event rates for infection and recovery
        """
        event_rate = [None] * params.number_events  # preallocate

        if type(params.b) == list:
            if self.species== Species.falciparum:
                lamx = params.b[self.species] * params.epsilonM[self.species] * (mozzie.mozzie_I_count[self.species] + (1-params.epsilonM[Species.vivax])*mozzie.mozzie_I_count[Species.mixed]) / params.human_population
            else:
                lamx = params.b[self.species] * params.epsilonM[self.species] * (mozzie.mozzie_I_count[self.species] + (1 - params.epsilonM[Species.falciparum]) * mozzie.mozzie_I_count[Species.mixed]) / params.human_population
        else:
            if self.species == Species.falciparum:
                lamx = params.b * params.epsilonM[self.species] * (mozzie.mozzie_I_count[self.species] + (1 - params.epsilonM[Species.vivax]) * mozzie.mozzie_I_count[Species.mixed]) / params.human_population
            else:
                lamx = params.b * params.epsilonM[self.species] * (mozzie.mozzie_I_count[self.species] + (1 - params.epsilonM[Species.falciparum]) *mozzie.mozzie_I_count[Species.mixed]) / params.human_population

        # 0: agent can be infected by mozzies with single or mixed infection
        event_rate[Transitions.S_inf] = lamx

        # 1: I -> A or death
        event_rate[Transitions.I_next] = params.sigma[self.species]

        event_rate[Transitions.I_treat] = params.c[self.species] * params.tau[self.species]

        # 3: A -> R or L
        event_rate[Transitions.A_recover] = params.alpha[self.species]

        # 2: I -> T or G
        # 4: A -> T or G
        if time >= params.mda_t1 and time < params.mda_t2:
            event_rate[Transitions.I_treat] = params.c[self.species] * params.tau[self.species] + params.eta[self.species] + params.etaFSAT[self.species] + params.etaMDA[self.species]

            event_rate[Transitions.A_treat] = params.eta[self.species] + params.etaFSAT[self.species] + params.etaMDA[self.species]
        else:
            if time>params.mda_t2:
                #make sure MDA every 6 months
                params.mda_t1 = params.mda_t1 + (365.25/2)
                params.mda_t2 = params.mda_t2 + (365.25/2)
            event_rate[Transitions.I_treat] = params.c[self.species] * params.tau[self.species] + params.eta[self.species] + params.etaFSAT[self.species]
            event_rate[Transitions.A_treat] = params.eta[self.species] + params.etaFSAT[self.species]

        # 5: R -> I or A
        event_rate[Transitions.R_inf] = params.r[self.species] * lamx

        # 6: R -> S
        event_rate[Transitions.R_S] = params.omega[self.species]

        # 7: L -> I or A
        event_rate[Transitions.L_inf] = params.r[self.species] * lamx

        # 8: L -> I or A
        event_rate[Transitions.L_relapse] = params.nu[self.species]

        # 9: L -> S
        event_rate[Transitions.L_S] = params.kappa[self.species]

        # 10: T -> ...
        event_rate[Transitions.T_done] = params.rho[self.species]

        # 11: G -> ...
        event_rate[Transitions.G_done] = params.psi[self.species]

        #event_rate[Transitions.mixed_inf] =
        return event_rate
