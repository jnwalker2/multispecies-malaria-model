from index_names import Compartments

class Pathogen(object):
    """
    the `state' of this Disease, wrt
    :param.current: current compartment.current (S, I, or R)
    :param.time: time of next event
    :param next: compartment transition to at time.time
    """
    __slots__ = ('current', 'time', 'next')

    def __init__(self, transition_time):
        """
        :param transition_time: time of transition from default allocation to susceptible class
        """
        self.current = Compartments.S
        self.time = transition_time  # int(time_end + 1)  # effectively infinity
        self.next = Compartments.S  # no default change from this compartment type


class Agent(object):
    """
    individuals in the human population
    """
    __slots__ = 'state', 'memory'

    def __init__(self, transition_time):
        """
        initialise Agent as per the Pathogen object
        :param transition_time: time of transition from default allocation to susceptible class
        """
        self.state = (Pathogen(transition_time=transition_time), Pathogen(transition_time=transition_time))  # todo: not hardcode this for 2 pathogens
        self.memory = ([], [])  # memory by species
