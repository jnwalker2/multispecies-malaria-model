from index_names import Species, Mozzie_labels
import math

class Mozzies(object):
    """
    track the Mozzies as a system of ODEs
    """
    def __init__(self, mozzie_initial_inf, human_initial_inf_comp_x_only, human_initial_mixed):
        """
        Merge hardcoded default values (from json file) with user-specified parameter values
        :param :
        """
        # required population level numbers
        self.mozzie_I_count = mozzie_initial_inf
        self.human_comp_x_only = human_initial_inf_comp_x_only
        self.human_compx_compy = human_initial_mixed

    def delta(self, time, params):
        """
        Sinusoidal forcing of the mozzie population
        :param time:
        :return:
        """
        forcing = params.delta0 * (1 - params.xi * math.cos(2 * math.pi * (time - params.phi) * params.time_day_step / 365.25+math.pi/2))
        return forcing

    def model(self, current_time, prev, params):
        """
        The system of ODEs
        :param prev: previous mozzie population by compartment
        :param current_time: current time
        :return: the updated mozzie population by compartment
        """
        # reassign to aid code readability
        X, Wf, Yf, Wv, Yv, Wfv, Zyf_wv, Zwf_yv, Yfv = prev
        total_mozzies = sum(prev)
        assert total_mozzies >= 0.0, ["Mozzie population is negative at t=" + str(current_time)]

        # forces of infection acting on mozzies -- truncated for truncated human model
        lambdaM_x = [None] * params.number_pathogens
        sum_mixed = [0] * params.number_pathogens  # using the for loop to calculate the species sum for the mixed force of infection
        lambdaM_total = [None] * params.number_pathogens  # for debugging
        # pf
        pf_only_sum = 0
        pf_inner_sum = 0
        pf_check_sum = 0
        for idx in range(4):
            pf_only_sum += params.epsilon_x[Species.falciparum][idx] * self.human_comp_x_only[Species.falciparum][idx]
            for idx_2 in range(4):
                pf_inner_sum += params.hat_epsilon_x[Species.falciparum][idx] * (1 - params.hat_epsilon_x[Species.vivax][idx_2]) * self.human_compx_compy[idx][idx_2]
                sum_mixed[Species.falciparum] += params.hat_epsilon_x[Species.falciparum][idx] * params.hat_epsilon_x[Species.vivax][idx_2] * self.human_compx_compy[idx][idx_2]
                pf_check_sum += params.hat_epsilon_x[Species.falciparum][idx] * self.human_compx_compy[idx][idx_2]
        if type(params.b) == list:
            lambdaM_x[Species.falciparum] = params.b[Species.falciparum] * (pf_only_sum + pf_inner_sum) / params.human_population
        else:
            lambdaM_x[Species.falciparum] = params.b * (pf_only_sum + pf_inner_sum) / params.human_population
        # pv
        pv_only_sum = 0
        pv_inner_sum = 0
        pv_check_sum = 0
        for idx in range(4):
            pv_only_sum += params.epsilon_x[Species.vivax][idx] * self.human_comp_x_only[Species.vivax][idx]
            for idx_2 in range(4):
                pv_inner_sum += params.hat_epsilon_x[Species.vivax][idx] * (1 - params.hat_epsilon_x[Species.falciparum][idx_2]) * self.human_compx_compy[idx_2][idx]
                sum_mixed[Species.vivax] += params.hat_epsilon_x[Species.vivax][idx] * params.hat_epsilon_x[Species.falciparum][idx_2] * self.human_compx_compy[idx_2][idx]
                pv_check_sum += params.hat_epsilon_x[Species.vivax][idx] * self.human_compx_compy[idx_2][idx]
        if type(params.b) == list:
            lambdaM_x[Species.vivax] = params.b[Species.vivax] * (pv_only_sum + pv_inner_sum) / params.human_population
        else:
            lambdaM_x[Species.vivax] = params.b * (pv_only_sum + pv_inner_sum) / params.human_population

        assert abs(sum_mixed[Species.falciparum] - sum_mixed[Species.vivax]) < 1e-6, "number of mixed infectious needs to be equal whichever species is indexed first, in Mozzies.model()"

        if type(params.b) == list:
            lambdaM_FV = (params.b[Species.falciparum] * sum_mixed[Species.falciparum] + params.b[Species.vivax] * sum_mixed[Species.vivax]) / (2 * params.human_population)  # hard-coded `number of pathogens = 2` here, since the code doesn't currently generalise anyway
            # for checking
            lambdaM_total[Species.falciparum] = params.b[Species.falciparum] * (pf_only_sum + pf_check_sum) / params.human_population
            lambdaM_total[Species.vivax] = params.b[Species.vivax] * (pv_only_sum + pv_check_sum) / params.human_population
        else:
            lambdaM_FV = params.b * (sum_mixed[Species.falciparum] + sum_mixed[Species.vivax]) / (2 * params.human_population)  # hard-coded `number of pathogens = 2` here, since the code doesn't currently generalise anyway
            # for checking
            lambdaM_total[Species.falciparum] = params.b * (pf_only_sum + pf_check_sum) / params.human_population
            lambdaM_total[Species.vivax] = params.b * (pv_only_sum + pv_check_sum) / params.human_population

        # time dependent parameters
        seasonal_forcing = self.delta(time=current_time, params=params)

        # set up array for change in mozzie pop
        mozzie_pop_change = [None] * 9

        # [0]: X :: Susceptible mozzies
        mozzie_pop_change[Mozzie_labels.X] = params.delta0 * total_mozzies - seasonal_forcing * X - (lambdaM_x[Species.falciparum] + lambdaM_x[Species.vivax] + lambdaM_FV) * X

        # [1]: Wf :: mozzies exposed to falciparum
        mozzie_pop_change[Mozzie_labels.Wf] = lambdaM_x[Species.falciparum] * X - (params.gamma[Species.falciparum] + seasonal_forcing) * Wf - \
                   params.vW[Species.falciparum] * (lambdaM_x[Species.vivax] + lambdaM_FV) * Wf + params.u[Species.falciparum] * lambdaM_x[Species.falciparum] * (params.vW[Species.vivax] * Wv + params.vY[Species.vivax] * Yv)

        # [2]: Yf :: mozzies infectious with falciparum
        mozzie_pop_change[Mozzie_labels.Yf] = params.gamma[Species.falciparum] * Wf - seasonal_forcing * Yf - params.vY[Species.falciparum] * (lambdaM_x[Species.vivax] + lambdaM_FV) * Yf

        # [3]: Wv :: mozzies exposed to vivax
        mozzie_pop_change[Mozzie_labels.Wv] = lambdaM_x[Species.vivax] * X - (params.gamma[Species.vivax] + seasonal_forcing) * Wv - \
                   params.vW[Species.vivax] * (lambdaM_x[Species.falciparum] + lambdaM_FV) * Wv + \
                   params.u[Species.vivax] * lambdaM_x[Species.vivax] * (params.vW[Species.falciparum] * Wf + params.vY[Species.falciparum] * Yf)

        # [4]: Yv :: mozzies infectious with vivax
        mozzie_pop_change[Mozzie_labels.Yv] = params.gamma[Species.vivax] * Wv - seasonal_forcing * Yv - params.vY[Species.vivax] * (lambdaM_x[Species.falciparum] + lambdaM_FV) * Yv

        # [5]: Wfv :: mozzies exposed to falciparum and vivax
        mozzie_pop_change[Mozzie_labels.Wfv] = lambdaM_FV * (X + params.vW[Species.falciparum] * Wf + params.vW[Species.vivax] * Wv) - \
                   (params.gamma[Species.falciparum] + params.gamma[Species.vivax] + seasonal_forcing) * Wfv + \
                   (1 - params.u[Species.falciparum]) * params.vW[Species.vivax] * lambdaM_x[Species.falciparum] * Wv + \
                   (1 - params.u[Species.vivax]) * params.vW[Species.falciparum] * lambdaM_x[Species.vivax] * Wf

        # [6]: Z_Yf_Wv :: mozzies infectious with falciparum and exposed to vivax
        mozzie_pop_change[Mozzie_labels.Z_yf_wv] = params.gamma[Species.falciparum] * Wfv - (params.gamma[Species.vivax] + seasonal_forcing) * Zyf_wv + lambdaM_FV * params.vY[Species.falciparum] * Yf + \
                   (1 - params.u[Species.vivax]) * params.vY[Species.falciparum] * lambdaM_x[Species.vivax] * Yf

        # [7]: Z_Wf_Yv :: mozzies exposed to falciparum and infectious with vivax
        mozzie_pop_change[Mozzie_labels.Z_wf_yv] = params.gamma[Species.vivax] * Wfv - (params.gamma[Species.falciparum] + seasonal_forcing) * Zwf_yv + lambdaM_FV * params.vY[Species.vivax] * Yv + \
                   (1 - params.u[Species.falciparum]) * params.vY[Species.vivax] * lambdaM_x[Species.falciparum] * Yv

        # [8]: Yfv :: mozzies infectious with falciparum and vivax
        mozzie_pop_change[Mozzie_labels.Yfv] = params.gamma[Species.falciparum] * Zwf_yv + params.gamma[Species.vivax] * Zyf_wv - seasonal_forcing * Yfv

        return mozzie_pop_change
