# -------------- PART 0: PYTHON PRELIM --------------

# Additional notes: 
# mosa.py evolve() function has been edited

# Import packages
import importlib
import os
import time
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
import pyvista as pv
import gc
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from math import inf
from numpy import random
from scipy.spatial import ConvexHull, Delaunay
from functions import *

# Name for text file to records stats
output_file = f"MosaStats.txt"


# -------------- PART 0a: CHOOSE CIRCUIT AND SET UP FOLDER --------------


# Choose circuit
circuit = input("Please enter name of the circuit: ")

# Import circuit config file
config = importlib.import_module(circuit)

# Define subfolder name to work in
folder_name = f"MOSA_{circuit}"

# Create folder if not yet exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Jump to folder
os.chdir(folder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")

# -------------- PART 0b: DEFINE DYNAMICAL SYSTEM --------------


# dx/dt
Equ1 = config.Equ1
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    alpha = params[0]
    n     = params[1]
    val0 = Equ1(x, alpha, n)
    return np.array([val0])

# Define initial time
t = 0.0

# Define number of steady states expected
numss = int(input("""
Do you expect 1 or 2 stable steady states in your search space? 
Please enter either 1 or 2: """))


# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------


# Load analytical sensitivity expressions
S_alpha_xss_analytic = config.S_alpha_xss_analytic
S_n_xss_analytic = config.S_n_xss_analytic


# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------


# Print prompt
print("""
Only two sensitivity functions are present:
0. |S_alpha_xss|
1. |S_n_xss|
MOSA will anneal this pair.
""")

# Choose pair of functions
choice1 = 0
choice2 = 1

# List of sensitivity function names
sensitivity_labels = [
    "|S_alpha_xss|",
    "|S_n_xss|"]

# Save function names for later use
label1 = sensitivity_labels[choice1]
label2 = sensitivity_labels[choice2]


# -------------- PART 0e: CHANGING DIRECTORIES --------------


# Define subfolder name to work in
subfolder_name = f"MOSA_sensfuncs_{choice1}_and_{choice2}"

# Create folder if not yet exist
if not os.path.exists(subfolder_name):
    os.makedirs(subfolder_name)

# Jump to folder
os.chdir(subfolder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")

# Record info about system
with open(output_file, "w") as file:
    file.write("--------------------------------------------\n")
    file.write("System information:\n")
    file.write("--------------------------------------------\n")
    file.write(f"Circuit choice: {circuit}\n")
    file.write(f"Number of steady states expected: {numss}\n")
    file.write(f"Sensitivity function 1: {label1}\n")
    file.write(f"Sensitivity function 2: {label2}\n")

# -------------- PART 0f: DEFINE FUNCTIONS --------------

# DEFINE FUNCTION TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def euclidean_distance(x1, x2):
    return abs(x1 - x2)

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS GIVEN AN INITIAL GUESS
def ssfinder(alpha_val,n_val):
        
    # -------------------------------------------------------------------------------------------------------------------
    # If we have one steady state                                                                                       #|
    if numss == 1:                                                                                                      #|
                                                                                                                        #|
        # Load initial guesses for solving which can be a function of a choice of alpha and n values                    #|
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)                                                 #|
                                                                                                                        #|
        # Define array of parameters                                                                                    #|
        params = np.array([alpha_val, n_val])                                                                           #|
                                                                                                                        #|
        # For each initial guess in the list of initial guesses we loaded                                               #|
        for InitGuess in InitGuesses:                                                                                   #|
                                                                                                                        #|
            # Get solution details                                                                                      #|
            output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)      #|
            xss = output                                                                                                #| If we inputted 1 
            fvec = infodict['fvec']                                                                                     #| for numss prompt
                                                                                                                        #|
            # Check if stable attractor point                                                                           #|
            delta = 1e-8                                                                                                #|
            dEqudx = (Equs(xss+delta, t, params)-Equs(xss, t, params))/delta                                            #|
            jac = np.array([[dEqudx]])                                                                                  #|
            eig = jac                                                                                                   #|
            instablility = np.real(eig) >= 0                                                                            #|
                                                                                                                        #|
            # Check if it is sufficiently large, has small residual, and successfully converges                         #|
            if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:                    #|
                # If so, it is a valid solution and we return it                                                        #|
                return xss                                                                                              #|
                                                                                                                        #|
        # If no valid solutions are found after trying all initial guesses                                              #|
        return float('nan')                                                                                             #|
                                                                                                                        #|
    # -------------------------------------------------------------------------------------------------------------------
        
    # -------------------------------------------------------------------------------------------------------------------
    # If we have two steady states                                                                                      #|
    if numss == 2:                                                                                                      #|
                                                                                                                        #|
        # Create an empty numpy array                                                                                   #|
        xss1 = np.array([])                                                                                             #|
        xss2 = np.array([])                                                                                             #|
                                                                                                                        #|
        # Load initial guesses which is a function of a choice of alpha and n values                                    #|
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)                                                 #|
                                                                                                                        #|
        # Define array of parameters                                                                                    #|
        params = np.array([alpha_val, n_val])                                                                           #|
                                                                                                                        #|
        # Initially we don't have a valid solution                                                                      #|
        is_valid = False                                                                                                #|
                                                                                                                        #|
        # Make a list to temporarily store the multiple valid solutions for each initial guess                          #|
        solutions = []                                                                                                  #|
                                                                                                                        #|
        # For each until you get one that gives a solution or you exhaust the list                                      #|
        for InitGuess in InitGuesses:                                                                                   #|
                                                                                                                        #|
            # Get solution details                                                                                      #|
            output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)      #|
            xss = output                                                                                                #|
            fvec = infodict['fvec']                                                                                     #|
                                                                                                                        #|
            # Check if stable attractor point                                                                           #| TO DO:
            delta = 1e-8                                                                                                #| If we inputted 2
            dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta                                #| for numss prompt.
            jac = np.array([[dEqudx]])                                                                                  #| This however needs
            eig = jac                                                                                                   #| us to have case of
            instablility = np.real(eig) >= 0                                                                            #| if numss == 1 and
                                                                                                                        #| if numss == 2 in
            # Check if it is sufficiently large, has small residual, and successfully converges                         #| fobj and the rest.
            if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:                    #|
                # If this is the first valid solution, just store it                                                    #|
                if len(solutions) == 0:                                                                                 #|
                    solutions.append(xss)                                                                               #|
                # If this is not the first valid solution found                                                         #|
                else:                                                                                                   #|
                    # Compare the new solution with the previous ones to see if it is far enough                        #|
                    if all(euclidean_distance(existing_x, xss) > DISTANCE_THRESHOLD for existing_x in solutions):       #|
                        # If it is far enough, store it and break out of the loop                                       #|
                        solutions.append(xss)                                                                           #|
                        break                                                                                           #|
                                                                                                                        #|
        # After looping through all the initial guesses                                                                 #|
        # If two distinct solutions found                                                                               #|
        if len(solutions) == 2:                                                                                         #|
            # Sort and store them                                                                                       #|
            solutions.sort()                                                                                            #|
            xss1 = np.append(xss1,solutions[0])                                                                         #|
            xss2 = np.append(xss2,solutions[1])                                                                         #|
        # If only one distinct solution found                                                                           #|
        elif len(solutions) == 1:                                                                                       #|
            # Store it twice                                                                                            #|
            xss1 = np.append(xss1,solutions[0])                                                                         #|
            xss2 = np.append(xss2,solutions[0])                                                                         #|
        # If no valid solutions found                                                                                   #|
        else:                                                                                                           #|
            # Store NaN                                                                                                 #|
            xss1 = np.append(xss1,float('nan'))                                                                         #|
            xss2 = np.append(xss2,float('nan'))                                                                         #|
        return xss1, xss2                                                                                               #|
    # -------------------------------------------------------------------------------------------------------------------
    
# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(xss_list, alpha_list, n_list, choice1, choice2):
    
    # Evaluate sensitivities
    S_alpha_xss = S_alpha_xss_analytic(xss_list, alpha_list, n_list)
    S_n_xss     = S_n_xss_analytic(xss_list, alpha_list, n_list)

    # Sensitivity dictionary
    sensitivities = {
        "S_alpha_xss": S_alpha_xss,
        "S_n_xss": S_n_xss}

    # Map indices to keys
    labels = {
        0: "S_alpha_xss",
        1: "S_n_xss"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]


# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):

    '''
    Uncomment the following line if need to see solution printed.
    It will look like this:
    Solution:  {'alpha': 3.239629898497018, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.7701015749115143, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.6542032278143664, 'n': 9.910685695594527}
    Solution:  {'alpha': 2.6542032278143664, 'n': 9.363644921265}
    Solution:  {'alpha': 3.0278948409846858, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.6451188083692183, 'n': 9.996303250351326}
    '''
    # print("Solution: ", solution)
	
	# Update parameter set
    alpha_val = solution["alpha"]
    n_val = solution["n"]

    # Create an empty numpy array
    xss_collect = np.array([])

    # Find steady states and store
    xss = ssfinder(alpha_val,n_val)
    xss_collect = np.append(xss_collect,xss)
    
    # Get sensitivity pair
    sens1, sens2 = senpair(xss_collect, solution["alpha"], solution["n"], choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    return ans1, ans2
    

# -------------- PART 1: GAUGING MOSA PARAMETERS --------------


# Record info
with open(output_file, "a") as file:
    file.write("--------------------------------------------\n")
    file.write("System probing to estimate MOSA run parameters:\n")
    file.write("--------------------------------------------\n")

# Sample alpha values
alpha_min = float(input("Please enter minimum alpha value: "))
alpha_max = float(input("Please enter maximum alpha value: "))
alpha_sampsize = int(input("Please enter the number of alpha samples: "))
alpha_samps = np.linspace(alpha_min, alpha_max, alpha_sampsize)

# Record info
with open(output_file, "a") as file:
    file.write(f"alpha values from {alpha_min} to {alpha_max} with {alpha_sampsize} linspaced samples\n")

# Sample n values
n_min = float(input("Please enter minimum n value: "))
n_max = float(input("Please enter maximum n value: "))
n_sampsize = int(input("Please enter the number of n samples: "))
n_samps = np.linspace(n_min, n_max, n_sampsize)

# Record info
with open(output_file, "a") as file:
    file.write(f"n values from {n_min} to {n_max} with {n_sampsize} linspaced samples\n")

# Create empty arrays to store corresponding values of xss and yss
xss_samps = np.array([])
sens1_samps = np.array([])
sens2_samps = np.array([])

# For each combination of parameters
for i in alpha_samps:
    for j in n_samps:
        
        # Get steady state value and store
        xss = ssfinder(i,j)
        xss_samps = np.append(xss_samps,xss)
        # Get corresponding sensitivities and store
        sens1, sens2 = senpair(xss, i, j, choice1, choice2)
        sens1_samps = np.append(sens1_samps,sens1)
        sens2_samps = np.append(sens2_samps,sens2)

# Get min and max of each sensitivity and print
sens1_samps_min = np.nanmin(sens1_samps)
sens2_samps_min = np.nanmin(sens2_samps)
sens1_samps_max = np.nanmax(sens1_samps)
sens2_samps_max = np.nanmax(sens2_samps)

# Record info
with open(output_file, "a") as file:
    file.write(f"Min sampled value of {label1}: {sens1_samps_min}\n")
    file.write(f"Min sampled value of {label2}: {sens2_samps_min}\n")
    file.write(f"Max sampled value of {label1}: {sens1_samps_max}\n")
    file.write(f"Max sampled value of {label2}: {sens2_samps_max}\n")

# Get MOSA energies
deltaE_sens1 = sens1_samps_max - sens1_samps_min
deltaE_sens2 = sens2_samps_max - sens2_samps_min
deltaE = np.linalg.norm([deltaE_sens1, deltaE_sens2])

# Record info
with open(output_file, "a") as file:
    file.write(f"Sampled energy difference in {label1}: {deltaE_sens1}\n")
    file.write(f"Sampled energy difference in {label2}: {deltaE_sens2}\n")
    file.write(f"Sampled cumulative energy difference: {deltaE}\n")

# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.9): "))
temp_hot = deltaE / np.log(1/probability_hot)

# Record info
with open(output_file, "a") as file:
    file.write(f"Chosen probability of transitioning to a higher energy state in hot run: {probability_hot}\n")
    file.write(f"Corresponding hot run tempertaure: {temp_hot}\n")
    file.write("(This temperature will be used to start the inital anneal.)")

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.01): "))
temp_cold = deltaE / np.log(1/probability_cold)

# Record info
with open(output_file, "a") as file:
    file.write(f"Chosen probability of transitioning to a higher energy state in cold run: {probability_cold}\n")
    file.write(f"Corresponding cold run tempertaure: {temp_cold}\n")
    file.write("(This temperature will be used to estimate when to end hot run. The actual finishing temperature from the hot run will used for the cold run.)\n")


# -------------- PART 2a: MOSA PREPARATIONS --------------


# Print prompts
print("Now preparing to MOSA...")
runs = int(input("Please enter number of MOSA runs you would like to complete (if in doubt enter 5): "))
iterations = int(input("Please enter number of random walks per run (if in doubt enter 100): "))

# Record info
with open(output_file, "a") as file:
    file.write("--------------------------------------------\n")
    file.write("MOSA run parameters:\n")
    file.write("--------------------------------------------\n")
    file.write(f"Chosen number of MOSA runs: {runs}\n")
    file.write(f"Chosen number of random walks per run: {iterations}\n")

# For each run
for run in range(runs):
    print(f"MOSA run number: {run+1}")
    
    # Record info
    with open(output_file, "a") as file:
        file.write(f"\n")
        file.write(f"MOSA RUN NUMBER {run+1}:\n")
    
    # Define lists to collect Pareto-optimal sensitivity and parameter values from each MOSA run
    pareto_Salpha = []
    pareto_Sn     = []
    pareto_alpha  = []
    pareto_n      = []
    
    # Delete archive and checkpoint json files at the start of each new run
    files_to_delete = ["archive.json", "checkpoint.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

	# -------------- PART 2b: ANNEAL TO GET PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Set random seed for MOSA
    random.seed(run)
    with open(output_file, "a") as file:
        file.write(f"\n")
        file.write(f"Random seed value for {run+1}: {random.random()}\n")
	
	# Initialisation of MOSA
    opt = mosa.Anneal()
    opt.archive_size = 10000
    opt.maximum_archive_rejections = opt.archive_size
    opt.population = {"alpha": (alpha_min, alpha_max), "n": (n_min, n_max)}
	
	# Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = iterations
    opt.temperature_decrease_factor = 0.95
    opt.number_of_temperatures = int(np.ceil(np.log(temp_cold / temp_hot) / np.log(opt.temperature_decrease_factor)))
    opt.number_of_solution_elements = {"alpha":1, "n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
    	
    # Hot run
    start_time = time.time()
    hotrun_stoppingtemp = opt.evolve(fobj)

    # Record info
    with open(output_file, "a") as file:
        file.write(f"\n")
        file.write(f"HOT RUN NO. {run+1}:\n")
        file.write(f"Hot run time: {time.time() - start_time} seconds\n")
        file.write(f"Hot run stopping temperature = cold run starting temperature: {hotrun_stoppingtemp}\n")
        file.write(f"Number of temperatures: {opt.number_of_temperatures}\n")
        file.write(f"Step scaling factor: {step_scaling}\n")
	
    # Cold run options
    opt.initial_temperature = hotrun_stoppingtemp
    opt.number_of_iterations = iterations
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"alpha":1,"n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Cold run
    start_time = time.time()
    coldrun_stoppingtemp = opt.evolve(fobj)
    print(f"Cold run time: {time.time() - start_time} seconds")
    
    # Record info
    with open(output_file, "a") as file:
        file.write(f"\n")
        file.write(f"COLD RUN NO. {run+1}:\n")
        file.write(f"Cold run time: {time.time() - start_time} seconds\n")
        file.write(f"Cold run stopping temperature: {coldrun_stoppingtemp}\n")
    
    # Output 
    start_time = time.time()
    pruned = opt.prunedominated()
    
    # Record info
    with open(output_file, "a") as file:
        file.write(f"\n")
        file.write(f"PRUNE NO. {run+1}:\n")
        file.write(f"Prune time: {time.time() - start_time} seconds\n")
	
	# -------------- PART 2c: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["alpha"] for solution in data["Solution"]])
    
    # Record info
    with open(output_file, "a") as file:
        file.write(f"Archive length after prune: {length}\n")
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_Salpha.append(dummy1)
        pareto_Sn.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'pruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
	
    # -------------- PART 2d: STORE AND PLOT CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract alpha and n values from the solutions
    alpha_values = [solution["alpha"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(alpha_values, n_values):
        pareto_alpha.append(dummy1)
        pareto_n.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(alpha_values, n_values)
    plt.xlabel('alpha')
    plt.ylabel('n')
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'pruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()

    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save S_a pareto values
    filename = f"pareto_Salpha_run{run+1}.npy"
    np.save(filename,pareto_Salpha)
    # Save S_n pareto values
    filename = f"pareto_Sn_run{run+1}.npy"
    np.save(filename,pareto_Sn)
    # Save a pareto values
    filename = f"pareto_alpha_run{run+1}.npy"
    np.save(filename,pareto_alpha)
    # Save n pareto values
    filename = f"pareto_n_run{run+1}.npy"
    np.save(filename,pareto_n)

# -------------- PART 2f: COMBINE PARETO DATA --------------

# Record text file prompt
with open(output_file, "a") as file:
    file.write("--------------------------------------------\n")
    file.write("Combining all individual run data:\n")
    file.write("--------------------------------------------\n")
    
# Combine Pareto Salpha data
pareto_Salpha_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_Salpha_run{run+1}.npy"
    pareto_Salpha = np.load(filename)  
    pareto_Salpha_combined = np.concatenate((pareto_Salpha_combined, pareto_Salpha))
    
# Save Pareto Salpha data
save_filename = "pareto_Salpha_combined.npy"
np.save(save_filename, pareto_Salpha_combined)

# Record length of Pareto Salpha data
length = len(pareto_Salpha_combined)
with open(output_file, "a") as file:
    file.write(f"Length of pareto_Salpha_combined: {length}\n")
    
# Free up memory
del pareto_Salpha_combined, pareto_Salpha
gc.collect()


# Combine Pareto Sn data
pareto_Sn_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_Sn_run{run+1}.npy"
    pareto_Sn = np.load(filename)  
    pareto_Sn_combined = np.concatenate((pareto_Sn_combined, pareto_Sn))
    
# Save Pareto Sn data
save_filename = "pareto_Sn_combined.npy"
np.save(save_filename, pareto_Sn_combined)

# Record length of Pareto Sn data
length = len(pareto_Sn_combined)
with open(output_file, "a") as file:
    file.write(f"Length of pareto_Sn_combined: {length}\n")
    
# Free up memory
del pareto_Sn_combined, pareto_Sn
gc.collect()


# Combine Pareto alpha data
pareto_alpha_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_alpha_run{run+1}.npy"
    pareto_alpha = np.load(filename)  
    pareto_alpha_combined = np.concatenate((pareto_alpha_combined, pareto_alpha))
    
# Save Pareto alpha data
save_filename = "pareto_alpha_combined.npy"
np.save(save_filename, pareto_alpha_combined)

# Record length of Pareto alpha data
length = len(pareto_alpha_combined)
with open(output_file, "a") as file:
    file.write(f"Length of pareto_alpha_combined: {length}\n")
    
# Free up memory
del pareto_alpha_combined, pareto_alpha
gc.collect()


# Combine Pareto n data
pareto_n_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_n_run{run+1}.npy"
    pareto_n = np.load(filename)  
    pareto_n_combined = np.concatenate((pareto_n_combined, pareto_n))
    
# Save Pareto n data
save_filename = "pareto_n_combined.npy"
np.save(save_filename, pareto_n_combined)

# Record length of Pareto n data
length = len(pareto_n_combined)
with open(output_file, "a") as file:
    file.write(f"Length of pareto_n_combined: {length}\n")
    
# Free up memory
del pareto_n_combined, pareto_n
gc.collect()


# -------------- PART 3: CUMULATIVE NEW PARETO OPTIMAL POINTS --------------


# 3a: LOAD DATA

filename = "pareto_Salpha_combined.npy"
pareto_Salpha_combined = np.load(filename)

filename = "pareto_Sn_combined.npy"
pareto_Sn_combined = np.load(filename)

filename = "pareto_alpha_combined.npy"
pareto_alpha_combined = np.load(filename)

filename = "pareto_n_combined.npy"
pareto_n_combined = np.load(filename)

# 3b: PLOT CUMULATIVE SENSITIVITY SPACE

plt.figure(figsize=(5,5))
plt.scatter(pareto_Salpha_combined, pareto_Sn_combined, s=10)
plt.xlabel(r'$|S_{a}(x_{ss})|$')
plt.ylabel(r'$|S_{n}(x_{ss})|$')
plt.grid(True)
plt.title(f'Cumulative Pareto Front from {runs} Runs')
plt.savefig(f'cumulative_pareto_sensitivities.png', dpi=300)
plt.close()

# 3c: PLOT CUMULATIVE PARAMETER SPACE

plt.figure(figsize=(5,5))
plt.scatter(pareto_alpha_combined, pareto_n_combined, s=10)
plt.xlabel(r'$a$')
plt.ylabel(r'$n$')
plt.grid(True)
plt.title(f'Corresponding Parameters from {runs} Runs')
plt.savefig(f'cumulative_pareto_parameters.png', dpi=300)
plt.close()


# -------------- PART 4: GETTING NEW REDUCED PARAMETER SPACE FOR GRID SEARCH --------------

# 4a: FIND RECTANGLE BOUNDS

# Define scattered points in 2D sensitivity space
points = np.array(list(zip(pareto_alpha_combined, pareto_n_combined)))

# Define rectangle to bound scatter
min_vals = np.min(points, axis=0)
max_vals = np.max(points, axis=0)

# Record info
with open(output_file, "a") as file:
    file.write("-------------------------------------------------\n")
    file.write(f"Bounds of new parameter space after {runs} runs:\n")
    file.write("-------------------------------------------------\n")
    file.write(f"alpha_min: {min_vals[0]}, alpha_max: {max_vals[0]}\n")
    file.write(f"n_min: {min_vals[1]}, n_max: {max_vals[1]}\n")

# Free up memory
del points
gc.collect()

# 4b: COMPARE OLD VS NEW PARAM SPACE AREA

# Volume of the bounding rectangular prism
new_param_area = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1])
# Volume of original parameter space
old_param_area = (alpha_max-alpha_min) * (n_max-n_min)
# Volume reduction as percentage
percentage = (new_param_area / old_param_area) * 100

# Record info
with open(output_file, "a") as file:
    file.write(f"New parameter space area: {new_param_area}\n")
    file.write(f"Old parameter space area: {old_param_area}\n")
    file.write(f"New parameter space is {percentage:.2f}% of original parameter space area.\n")

# 4c: SAMPLE WITHIN NEW PARAM SPACE WITH SAME DENSITY AS ORIGINAL GRID SEARCH

# Create a grid of evenly spaced points from old parameter space
alpha_numofpoints = 100 #5000
n_numofpoints = 100 #5000
a_vals = np.linspace(alpha_min,alpha_max,alpha_numofpoints)
n_vals = np.linspace(n_min,n_max,n_numofpoints)
grid_x, grid_y = np.meshgrid(a_vals,n_vals)
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Define Bounding Rectangle Manually
min_x, min_y = min_vals
max_x, max_y = max_vals
bounding_box = np.array([
    [min_x, min_y],  # Bottom-left
    [max_x, min_y],  # Bottom-right
    [max_x, max_y],  # Top-right
    [min_x, max_y],  # Top-left
    [min_x, min_y]   # Close the rectangle
])

# Filter points inside the bounding rectangle
inside_rect_mask = (
    (grid_points[:, 0] >= min_x) & (grid_points[:, 0] <= max_x) &
    (grid_points[:, 1] >= min_y) & (grid_points[:, 1] <= max_y)
)
inside_rect_points = grid_points[inside_rect_mask]

# Save data
np.save("inside_points.npy", inside_rect_points)

# Record info
with open(output_file, "a") as file:
    file.write(f"Alpha line density: {alpha_numofpoints / (alpha_max-alpha_min)} points per unit alpha \n")
    file.write(f"n line density: {n_numofpoints / (n_max-n_min)} points per unit n \n")
    file.write(f"Area density: {(alpha_numofpoints * n_numofpoints) / ((alpha_max-alpha_min) * (n_max-n_min))} points per unit area \n")
    file.write(f"Number of points: {np.shape(inside_rect_points)[0]}\n")

# 4d: PLOT BOUNDING BOX

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the bounding rectangle
bounding_box = np.array([
    [min_x, min_y],  # Bottom-left
    [max_x, min_y],  # Bottom-right
    [max_x, max_y],  # Top-right
    [min_x, max_y],  # Top-left
    [min_x, min_y]   # Close the rectangle
])
ax.plot(bounding_box[:, 0], bounding_box[:, 1], color="red", linewidth=2, label="Bounding Rectangle")

# Cosmetics 
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$n$")
ax.set_title("Bounding Rectangle and Sampled Points")
ax.set_xlim([alpha_min,alpha_max])
ax.set_ylim([n_min,n_max])
ax.legend()

# Save and close
plt.savefig(f'new_vs_old_paramspace.png', dpi=300)
plt.close()


# -------------- PART 5: GRID SEARCH --------------


# 5A: IMPORT PACKAGES...

from tqdm import tqdm
from paretoset import paretoset
from joblib import Parallel, delayed

# 5B: SOLVE FOR X STEADY STATE VALUES...

# Print prompt
print("Solving for x steady states in the new reduced parameter space...")

# Get number of rows in the ParamPolygon
rows = inside_rect_points.shape[0]

# Create empty arrays to store x steady states
xssPolygon = np.empty((rows, 1))

# Define function to solve for steady states in parallel
def solve_steady_state(rownum, ParamPolygon):
    
    alpha_val = ParamPolygon[rownum][0]
    n_val = ParamPolygon[rownum][1]
    
    xss = ssfinder(alpha_val,n_val)
    return xss, rownum

# Parallel processing to solve steady states
results = Parallel(n_jobs=-1)(
    delayed(solve_steady_state)(rownum, inside_rect_points)
    for rownum in range(rows))

# Process results and store them in the polyhedron arrays
for xss, rownum in results:
    xssPolygon[rownum] = xss

# Save arrays
np.savez('PostMOSA_EquilibriumPolygons.npz', xssPolygon=xssPolygon) 


# 5C: OBTAIN TABLE OF SENSITIVITIES...

# Print prompt
print("Obtaining sensitivity values in the new reduced parameter space...")

# We want to get the following array
#  -------------------------
# | S_{a}(xss) | S_{n}(xss) |
# |      #     |      #     |
# |      #     |      #     |
# |      #     |      #     |
# |      #     |      #     |
#  -------------------------

def compute_sensitivities(rownum, ParamPolygon, xssPolygon):
    alpha_val = ParamPolygon[rownum, 0]
    n_val = ParamPolygon[rownum, 1]

    xss_val = xssPolygon[rownum]

    return np.array([
        S_alpha_xss_analytic(xss_val, alpha_val, n_val),
        S_n_xss_analytic(xss_val, alpha_val, n_val)])

# Parallel processing for sensitivity calculations
sensitivity_results = Parallel(n_jobs=-1)(
    delayed(compute_sensitivities)(rownum, inside_rect_points, xssPolygon)
    for rownum in range(rows))

# Collect results (the name SenPolygons is plural because in the same polygon parameter region, we have multiple sensitivity values)
SenPolygons = np.array(sensitivity_results).squeeze()

# Save table
np.save('PostMOSA_SensitivityPolygons.npy', SenPolygons)

# Free up memory
del xssPolygon
gc.collect()


# 5D: MOO...

# Print prompt
print("MOOing...")

# Pareto minimisation will think NaNs are minimum. Replace NaNs with infinities.
SenPolygons = np.where(np.isnan(SenPolygons), np.inf, SenPolygons)
mask = paretoset(SenPolygons, sense=["min", "min"])
pareto_Sens = SenPolygons[mask]
pareto_Params = inside_rect_points[mask]

# Saving
np.save('ARneg_ParetoMask.npy', mask)
np.save('ARneg_SensitivityPareto.npy', pareto_Sens)
np.save('ARneg_ParamPareto.npy', pareto_Params)

# Free up memory
del inside_rect_points, SenPolygons, mask
gc.collect()


# 5E: PLOT PARETO FRONT AND CORRESPONDING PARAMETERS...

# Print prompt
print("Plotting Pareto front and Pareto optimal parameters...")

# Make plot
fig, axes = plt.subplots(1, 2, figsize=(6,3), constrained_layout=True)
axes[0].scatter(   pareto_Sens[:,0],   pareto_Sens[:,1],   s=10)
axes[0].set_xlabel(label1)
axes[0].set_ylabel(label2)
axes[0].set_title(r'Pareto front')
axes[1].scatter(   pareto_Params[:,0],   pareto_Params[:,1],   s=10)
axes[1].set_xlabel(r'$\alpha$')
axes[1].set_ylabel(r'$n$')
axes[1].set_title(r'Pareto optimal parameters')
plt.savefig('PostMOSA_ParetoPlot.png', dpi=300)
plt.close()















# -------------- PART 6: COMPARE EVOLUTION OF POST-MOSA GRID SEARCHED PARETO FRONT AS A FUNCTION OF NUMBER OF MOSA RUNS --------------

# 6A: EVOLUTION OF GRID SEARCH SPACE WITH INCREASING RUNS

# Define function that returns a 2D numpy array containing rows from arr2 that are not in arr1, where arr1 and arr2 are 2D NumPy arrays
def rows_not_in_arr1(arr1, arr2):
    dtype = [('f0', arr2.dtype), ('f1', arr2.dtype)]
    arr1_struct = arr1.view(dtype)
    arr2_struct = arr2.view(dtype)
    unique_rows = np.setdiff1d(arr2_struct, arr1_struct).view(arr2.dtype).reshape(-1, arr2.shape[1])
    return unique_rows

# Define empty list to accumulate parameter space search points as number of MOSA runs increase
accumulated_searchpoints = np.empty((0, 2))

# For each run in our total number of runs
for run in range(1, runs+1):
	
	# Load the parameter values that were responsible for that run's Pareto front
    pareto_alpha = np.load(f"pareto_alpha_run{run}.npy", allow_pickle=True)
    pareto_n = np.load(f"pareto_n_run{run}.npy", allow_pickle=True)
    
    # Create a rectangular bounding box mask that encloses those paramter points
    new_rect_mask = (
    (grid_points[:, 0] >= min(pareto_alpha)) & (grid_points[:, 0] <= max(pareto_alpha)) &
    (grid_points[:, 1] >= min(pareto_n)) & (grid_points[:, 1] <= max(pareto_n))
    )
    
    # Filter to get the parameter space points inside that mask
    grid_points_run = grid_points[new_rect_mask]
    
    # Further filter to get the points that are not already in our list of accumulated parameter space search points
    new_searchpoints = rows_not_in_arr1(accumulated_searchpoints, grid_points_run)
    
    # Store these new points into our list of accumulated parameter search space points
    accumulated_searchpoints = np.vstack((accumulated_searchpoints, new_searchpoints))
	
	# Save the new parameter space search points for current run number and accumulated parameter space search points up to current run number
    np.save(f"new_searchpoints_run1to{run}.npy", new_searchpoints)
    np.save(f"accumulated_searchpoints_run1to{run}.npy", accumulated_searchpoints)
    
# 6B: EVOLUTION OF STEADY STATES WITH INCREASING RUNS

# Define empty list to accumulate steady state solutions as number of MOSA runs increase
accumulated_xss = np.empty((0, 1))

# For each run in our total number of runs
for run in range(1, runs+1):
	
	# Load the new parameter space search points of that run
    new_searchpoints = np.load(f"new_searchpoints_run1to{run}.npy", allow_pickle=True)
		
	# Get number of rows to know how many points we need to solve for steady state at
    rows = new_searchpoints.shape[0]
	
	# Create empty arrays to store x steady states
    new_xss = np.empty((rows, 1))
	
	# Parallel processing to solve steady states
    results = Parallel(n_jobs=-1)(
		delayed(solve_steady_state)(rownum, new_searchpoints)
		for rownum in range(rows))
		
	# Store results in empty array
    for xss, rownum in results:
        new_xss[rownum] = xss
	
	# Accumulate steady states in the global list we made
    accumulated_xss = np.vstack((accumulated_xss,new_xss))
	
	# Save the new steady state data for current run number and accumulated steady state data up to current run number
    np.save(f"new_xss_run1to{run}.npy", new_xss)
    np.save(f"accumulated_xss_run1to{run}.npy", accumulated_xss)
	
# 6C: EVOLUTION OF SENSITIVITY FUNCTION PAIRS WITH INCREASING RUNS

# Define empty list to accumulate steady state solutions as number of MOSA runs increase
accumulated_SenPairData = np.empty((0, 2))

# For each run in our total number of runs
for run in range(1, runs+1):

	# Load the new parameter space search points and new steady state data of that run
    new_searchpoints = np.load(f"new_searchpoints_run1to{run}.npy", allow_pickle=True)
    new_xss = np.load(f"new_xss_run1to{run}.npy", allow_pickle=True)
	
	# FUCK
    rows = np.shape(new_xss)[0]
	
	# Parallel processing for sensitivity calculations
    sensitivity_results = Parallel(n_jobs=-1)(
		delayed(compute_sensitivities)(rownum, new_searchpoints, new_xss)
		for rownum in range(rows))
		
	# Collect results 
    new_SenPairData = np.array(sensitivity_results).squeeze()
    
    print("Shape new_Senpairdata: ", np.shape(new_SenPairData))
    print("new_Senpairdata: ", new_SenPairData)
    
    print("Shape accumulated_SenPairData: ", np.shape(accumulated_SenPairData))
    
    # Store these new points into our list of accumulated parameter search space points
    accumulated_SenPairData = np.vstack((accumulated_SenPairData, new_SenPairData))
	
	# Save the new steady state data for current run number and accumulated steady state data up to current run number
    np.save(f"new_SenPairData_run1to{run}.npy", new_SenPairData)
    np.save(f"accumulated_SenPairData_run1to{run}.npy", accumulated_SenPairData)

	
# 6D: EVOLUTION OF PARETO FRONT WITH INCREASING RUNS

# Define empty list to accumulate steady state solutions as number of MOSA runs increase
accumulated_ParetoSens = np.empty((0, 2))
accumulated_ParetoParams = np.empty((0, 2))

# For each run in our total number of runs
for run in range(1, runs+1):

	# Load the the accumulated sensitivity pair data up to current run
    accumulated_SenPairData = np.load(f"accumulated_SenPairData_run1to{run}.npy", allow_pickle=True)
    accumulated_searchpoints = np.load(f"accumulated_searchpoints_run1to{run}.npy", allow_pickle=True)
	
	# Pareto minimisation will think NaNs are minimum. Replace NaNs with infinities.
    accumulated_SenPairData = np.where(np.isnan(accumulated_SenPairData), np.inf, accumulated_SenPairData)
    
    mask = paretoset(accumulated_SenPairData, sense=["min", "min"])
    pareto_Sens = accumulated_SenPairData[mask]
    pareto_Params = accumulated_searchpoints[mask]
    
    accumulated_ParetoSens = np.vstack((accumulated_ParetoSens, pareto_Sens))
    accumulated_ParetoParams = np.vstack((accumulated_ParetoParams, pareto_Params))
	
	# Save arrays
    np.save(f"accumulated_ParetoSens_run1to{run}.npy", accumulated_ParetoSens)
    np.save(f"accumulated_ParetoParams_run1to{run}.npy", accumulated_ParetoParams)
    
# 6E: PLOT EVOLUTION OF PARETO FRONT WITH INCREASING RUNS

# Assign colors for different runs
colors = plt.cm.viridis(np.linspace(0, 1, runs))

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Loop over runs with indexing
for i, run in enumerate(range(1, runs + 1)):

    # Load data for the current run
    accumulated_ParetoSens = np.load(f"accumulated_ParetoSens_run1to{run + 1}.npy")
    accumulated_ParetoParams = np.load(f"accumulated_ParetoParams_run1to{run + 1}.npy")
    
    # Plot Pareto front
    ax1.scatter(accumulated_ParetoSens[:, 0],
                accumulated_ParetoSens[:, 1],
                s=2,
                color=colors[i],
                label=f"{run + 1} runs")
    
    # Plot corresponding parameters
    ax2.scatter(accumulated_ParetoParams[:, 0],
                accumulated_ParetoParams[:, 1],
                s=2,
                color=colors[i],
                label=f"{run + 1} runs")

# Set labels, titles, and legends for the first subplot
ax1.set_xlabel(label1)
ax1.set_ylabel(label2)
ax1.set_title("Pareto front")
ax1.legend()

# Set labels, titles, and legends for the second subplot
ax2.set_xlabel(alpha)
ax2.set_ylabel(n)
ax2.set_title("Corresponding parameters")
ax2.legend()

# Set a main title for the entire figure
fig.suptitle("Evolution of Pareto Front as Number of MOSA Runs Increase")

# Save and show the plot
plt.savefig("paretofront_evolution.png", dpi=300)
plt.show()
