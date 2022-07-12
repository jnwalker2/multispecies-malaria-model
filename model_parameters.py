from index_names import Species, Compartments
import json
import numpy as np

class model_parameters(object):
    def __init__(self, **kwargs):
        """
        Merge hardcoded default values (from json file) with user-specified parameter values
        :param kwargs:
        """
        # simulation parameters
        self.number_events = 12
        self.number_repeats = 4
        self.number_pathogens = 2
        self.number_compartments = 7

        # set time related parameters
        self.time_day_start = 0
        self.time_day_end = 365.25 * 10
        if 'time_day_step' in kwargs:
            self.time_day_step = kwargs['time_day_step']
        else:
            self.time_day_step =1.0  # looks like this needs to be about ~0.1 for vivax: notable differences between stochastic and deterministic solutions if 0.5, less but still some with 0.2
        self.time_vec = np.arange(start=self.time_day_start, stop=self.time_day_end, step=self.time_day_step)

        # convert time to units of time_day_step
        self.time_start = int(round(self.time_day_start / self.time_day_step))
        self.time_end = int(round(self.time_day_end / self.time_day_step))

        # calculate time vectors needed for deterministic model
        self.t_det = np.arange(start=self.time_start, stop=self.time_end, step=0.5 / self.time_day_step)  # don't need to solve the odes with such a fine timestep
        self.time_vec_det = self.t_det * self.time_day_step

        self.human_population = 1000
        self.mozzie_human_pop_ratio = 3  # i.e. number of mosquitoes for each human
        self.period = 365.25

        # transmission model parameters

        # update default values with preference to user-specified ones
        args, self.param_mins, self.param_maxes = self.defaults()
        args.update(kwargs)

        for key, value in args.items():
            self.__setattr__(key, value)
        self.param_mins['period'] = self.period - 0.01 * self.period
        self.param_maxes['period'] = self.period + 0.01 * self.period

        self.mozzie_pop = self.human_population * self.mozzie_human_pop_ratio  # not sure if this belongs here or elsewhere

        # setting the falciparum values to zero to prevent the `L` compartment being used
        not_falciparum_params = ['kappa', 'nu', 'pA', 'ph', 'pL', 'pP']
        for name in not_falciparum_params:
            self.__setattr__(name, [0.0, self.__getattribute__(name)])
            self.param_mins[name] = [0.0, self.param_mins[name]]
            self.param_maxes[name] = [0.0, self.param_maxes[name]]

        self.calculate_interactions()  # make sure default interactions are calculated

        # calculate new transmission probabilities
        self.epsilon_x = [None] * self.number_pathogens
        self.hat_epsilon_x = [None] * self.number_pathogens
        for sp in [Species.falciparum, Species.vivax]:  # or range(number_pathogens) if this doesn't work
            self.epsilon_x[sp] = [self.epsilonH[sp] * self.zetaI[sp], self.epsilonH[sp] * self.zetaA[sp], self.epsilonH[sp] * self.zetaT[sp], self.epsilonH[sp] * self.zetaG[sp]]
            self.hat_epsilon_x[sp] = [self.hatepsilonH[sp] * self.zetaI[sp], self.hatepsilonH[sp] * self.zetaA[sp], self.hatepsilonH[sp] * self.zetaT[sp], self.hatepsilonH[sp] * self.zetaG[sp]]

        if type(self.delta0) == list:
            self.delta0 = sum(self.delta0) / len(self.delta0)  # use arithematic average

    def defaults(self, filename=r"parameter_ranges.json"):
        """
        Pull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return:
        """
        param_min = dict()
        param_max = dict()
        with open(filename) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            # if this is a rate, multiply by `time_day_step`
            _tmp = value["exp"]
            _tmp_min = value["min"]
            _tmp_max = value["max"]

            if value["units"] == "per day":
                if type(_tmp) == list:
                    _tmp = [_tmp[idx] * self.time_day_step for idx in range(len(_tmp))]
                    _tmp_min = [_tmp_min[idx] * self.time_day_step for idx in range(len(_tmp_min))]
                    _tmp_max = [_tmp_max[idx] * self.time_day_step for idx in range(len(_tmp_max))]
                else:
                    _tmp = _tmp * self.time_day_step
                    _tmp_min = _tmp_min * self.time_day_step
                    _tmp_max = _tmp_max * self.time_day_step
            elif value["units"] == "day":
                if type(_tmp) == list:
                    _tmp = [_tmp[idx] / self.time_day_step for idx in range(len(_tmp))]
                    _tmp_min = [_tmp_min[idx] / self.time_day_step for idx in range(len(_tmp_min))]
                    _tmp_max = [_tmp_max[idx] / self.time_day_step for idx in range(len(_tmp_max))]
                else:
                    _tmp = _tmp / self.time_day_step
                    _tmp_min = _tmp_min / self.time_day_step
                    _tmp_max = _tmp_max / self.time_day_step

            json_data[key] = _tmp
            param_min[key] = _tmp_min
            param_max[key] = _tmp_max
        return json_data, param_min, param_max

    def interactions_off(self):
        """
        Set for the interaction parameters between falciparum and vivax such that there is none.
        :return:
        """
        interaction_params = {"a": [1.0, 1.0], "g": 1.0, "fv": 0.0, "ell": 1.0, "pLam": [1.0, 1.0],
                              "zf": 1.0, "zv": 1.0, "pZf": 0.0, "pZv": 0.0,
                              "qT": [1.0, 1.0], "qG": [1.0, 1.0], "qrho": 1.0, "qpsi": 1.0,
                              "hv": 1.0,
                              "s": [1.0, 1.0], "j": [1.0, 1.0], "k": [1.0, 1.0], "n": [1.0, 1.0], "u": [0.0, 0.0], "vW": [1.0, 1.0], "vY": [1.0, 1.0], "flag_entangled_treatment": False, "flag_triggering_by_pf": False}

        return interaction_params

    def interactions_on(self):
        """
        Set for the interaction parameters between falciparum and vivax such that there is both entanglement and triggering.
        :return:
        """
        interaction_params = {"a": [1.0, 1.0], "g": 1.0, "fv": 0.0, "ell": 1.0, "pLam": [1.0, 1.0],
                              "zf": 3.5, "zv": 1.0, "pZf": 0.0, "pZv": 0.0,
                              "qT": [1.0, 1.0], "qG": [1.0, 1.0], "qrho": 1.0, "qpsi": 1.0,
                              "hv": 1.0,
                              "s": [1.0, 1.0], "j": [1.0, 1.0], "k": [1.0, 1.0], "n": [1.0, 1.0], "u": [0.0, 0.0], "vW": [1.0, 1.0], "vY": [1.0, 1.0], "flag_entangled_treatment": True, "flag_triggering_by_pf": True}

        return interaction_params

    def calculate_interactions(self, **interaction_params):

        interaction_args = self.interactions_on() # turn on entanglement and triggering by pf
        interaction_args.update(interaction_params)

        # calculate new args
        new_args = dict()

        # RBC competition within-humans
        new_args['hatepsilonH'] = [a * b for a, b in
                                   zip(interaction_args['pLam'], self.epsilonH)]  # affects transmission to mozzies

        # competition between species for the vector
        new_args['u'] = interaction_args['u']
        new_args['vW'] = interaction_args['vW']
        new_args['vY'] = interaction_args['vY']

        interaction_args.update(new_args)

        for key, value in interaction_args.items():
            self.__setattr__(key, value)

    def initial_conditions(self, num_cases=[10, 10]):

        # initial condition
        if len(num_cases) < 2:
            human_initial_inf = [10, 10]
            print('Warning: insufficient number of initial case counts in `model_parameters.initial_conditions`')
        else:
            human_initial_inf = num_cases
        human_initial_mixed_only = 0  # number mixed infections at t=0
        mozzie_initial_inf = [self.mozzie_human_pop_ratio * human_initial_inf[0], self.mozzie_human_pop_ratio * human_initial_inf[1], 0]  # [falciparum-only, vivax-only, mixed]
        assert human_initial_mixed_only <= human_initial_inf[Species.falciparum] and human_initial_mixed_only <= \
               human_initial_inf[Species.vivax]

        # use the above to determine compartment counts
        initial_mozzie = [self.mozzie_pop - sum(mozzie_initial_inf), 0.0, mozzie_initial_inf[Species.falciparum], 0.0, mozzie_initial_inf[Species.vivax], 0.0, 0.0, 0.0, mozzie_initial_inf[Species.mixed]]

        human_initial_inf_comp_x_only = [[human_initial_inf[Species.falciparum] - human_initial_mixed_only, 0, 0, 0],
                                         [human_initial_inf[Species.vivax] - human_initial_mixed_only, 0, 0, 0]]
        human_initial_mixed_all = [[human_initial_mixed_only, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]
        initial_human_population_counts = [[self.human_population - human_initial_inf[Species.falciparum],
                                            human_initial_inf[Species.falciparum], 0, 0, 0, 0, 0, 0],
                                           [self.human_population - human_initial_inf[Species.vivax],
                                            human_initial_inf[Species.vivax], 0, 0, 0, 0, 0, 0]]  # last one is for `just_died`

        # identifying population counts for all compartment combos
        initial_human_combo_counts = [[0 for i in range(self.number_compartments)] for j in range(self.number_compartments)]
        # setting up no mixed infections ICs
        initial_human_combo_counts[Compartments.S] = initial_human_population_counts[Species.vivax][:-1].copy()
        for idx in range(self.number_compartments):
            initial_human_combo_counts[idx][0] = initial_human_population_counts[Species.falciparum][idx]
        # now fix S count
        initial_human_combo_counts[0][0] = int(self.human_population - sum(human_initial_inf))

        return human_initial_inf, human_initial_mixed_only, mozzie_initial_inf, human_initial_inf_comp_x_only, human_initial_mixed_all, initial_human_population_counts, initial_mozzie, initial_human_combo_counts

class model_ranges(object):
    def __init__(self, **kwargs):

        if 'time_day_step' in kwargs:
            self.time_day_step = kwargs['time_day_step']
        else:
            self.time_day_step = 1.0  # looks like this needs to be about ~0.1 for vivax: notable differences between stochastic and deterministic solutions if 0.5, less but still some with 0.2

        # update default values with preference to user-specified ones
        args = self.defaults()
        args.update(kwargs)

        for key, value in args.items():
            self.__setattr__(key, value)

        # setting the falciparum values to zero to prevent the `L` compartment being used
        self.kappa = [0.0, self.kappa]
        self.nu = [0.0, self.nu]
        self.pA = [0.0, self.pA]
        self.ph = [0.0, self.ph]
        self.pL = [0.0, self.pL]
        self.pP = [0.0, self.pP]
        # setting all malaria induced deaths of the other species to zero and get results matching deteterministic model -- for debugging only!
        self.pI = [0.0, 0.0]
        self.pT = [0.0, 0.0]
        self.pG = [0.0, 0.0]

    def defaults(self, filename=r"parameter_ranges.json"):
        """
        Pull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return:
        """
        with open(filename) as json_file:
            json_data = json.load(json_file)

        for key, value in json_data.items():
            # if this is a rate, multiply by `time_day_step`
            _tmp = value["exp"]
            _tmp_max = value["max"]
            _tmp_min = value["min"]
            if type(_tmp) == list:
                _tmp_var = [0.01 * _tmp[idx] for idx in range(len(_tmp))] #[max(max(_tmp_max[idx]-_tmp[idx], _tmp[idx]-_tmp_min[idx]), 0.01 * _tmp[idx]) for idx in range(len(_tmp))]
            else:
                _tmp_var = 0.01 * _tmp #max(max(_tmp_max - _tmp, _tmp - _tmp_min), 0.01 * _tmp)

            if value["units"] == "per day":
                if type(_tmp) == list:
                    _tmp_var = [_tmp_var[idx] * self.time_day_step for idx in range(len(_tmp_var))]
                else:
                    _tmp_var = _tmp_var * self.time_day_step
            json_data[key] = _tmp_var
        return json_data
