# multispecies-malaria-model
Code to support the manuscript titled "A model for malaria treatment evaluation
in the presence of multiple species".

# Code authorship
Major initial model development and implementation were contributed by Roslyn Hickson.

The sensetivity analysis, incorporation of some interactions and general edits throughout were contributed by Edmond Chang.

Preparation of code for scenario modelling, incorporation of population treatment strategies (such as mass drug administration) and some interactions were contributed by Camelia Walker. 

# Usage
To run the model, put all files in this repository (including **stored/**) in
the same directory. Then run **do_simulations_flexiby.py**.

## Description of scripts
### do_simulations_flexibly.py
The only script that needs to be run directly. Calls the other scripts. Iterates
over treatment scenario and coverage scenario combinations. Treatment scenarios
in pN_vec; coverage scenarios in c_vec.

Values between lines 41-55 can be changed by the user. prov_list_epidemics
specifies which locations in **local_params.json** that a
simulation will be run for.

### H_abm_Mcomixing.py
Contains majority of code for implementation of the model. Initialises, updates,
and records states of agents. Also initialises, updates, and records various
counters and quantities of interest. Output files generated here.  

run_me in Run_Simulations calls other functions in Run_Simulations to run a
single simulation.  
use_calibrated_params updates parameters defined in
**local_params.json**. That is, parameter values in
**local_params.json** override parameter values defined in
**parameter_ranges.json**.  
do_iterate implements the running of multiple iterations. Includes switch for
parallelisation. Stores the outputs from the multiple iterations and generates
output files.

### disease_model.py
Logic for transitions between states of human agents. Also tracks population
level aggregates for quantities of interest e.g. deaths, treatments.

transition_table calculates next state given current state.  
triggering implements logic for triggering of P. vivax relapse by P. falciparum
infection.   
update implements transition to next state according to transition_table.  
infect_me implements infection event and transition to infected state.
stochastic_sir_event_rates calculates parameters that determine rates in the
functions above.

### model_parameters.py
Defines simulation parameters such as number of repeats and duration of
simulation. Sets default values of model parameters using
**parameter_ranges.json**. Also calculates interactions for entanglement and
triggering by pf.

### human_agents.py
Initialises objects for individual human agent.

### index_names.py
Indices for various states.  

### mosquito_model.py
Implementation of ODE model for mosquitos.

### parameter_ranges.py
Defines default values of parameters. Also includes additional information such as
parameter value ranges, parameter description, and source of value.

### stored/local_params.json
Defines location specific values of certain parameters. Any parameter values set
in **local_params.json** will override default parameter value
defined in **parameter_ranges.py**.
