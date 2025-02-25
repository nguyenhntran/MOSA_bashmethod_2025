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

# Import self made functions
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