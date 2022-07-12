from enum import IntEnum

class Compartments(IntEnum):
    """
    for speed ups whilst maintaining readability of code
    """
    S = 0
    I = 1
    A = 2
    R = 3
    L = 4
    T = 5
    G = 6
    just_died = 7  # use this to identify need to run `process_death`
    dead = 8

class Mozzie_labels(IntEnum):
    X = 0
    Wf = 1
    Yf = 2
    Wv = 3
    Yv = 4
    Wfv = 5
    Z_yf_wv = 6
    Z_wf_yv = 7
    Yfv = 8

class Species(IntEnum):
    falciparum = 0
    vivax = 1
    mixed = 2

class Transitions(IntEnum):
    """
    using notation `from`_`to`
    """
    S_inf = 0  # this is rate from S->I + S->A (i.e. then probability determines which)
    I_next = 1  # A or S loss of clinical symptoms and /or detectability
    I_treat = 2  # to T or G
    A_recover = 3  # to R or L
    A_treat = 4  # to T or G (MDA only)
    R_inf = 5   # to I or A
    R_S = 6  # waning immunity
    L_inf = 7  # to I or A
    L_relapse = 8  # to I or A
    L_S = 9  # hypnozoites all dead/absorbed/...
    T_done = 10  # no longer infectious from T
    G_done = 11  # no longer infectious from G
