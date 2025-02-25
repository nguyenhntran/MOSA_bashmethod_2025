import numpy as np

# ---------------------------------------------------------------------------------------------------
    
# DEFINE FUNCTION TO EVALUATE VECTOR FIELD
def Equs(P, t, params):
    x = P[0]
    alpha = params[0]
    n     = params[1]
    val0 = Equ1(x, alpha, n)
    return np.array([val0])

# ---------------------------------------------------------------------------------------------------

# DEFINE FUNCTION TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def euclidean_distance(x1, x2):
    return abs(x1 - x2)

# ---------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------

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
    
# ---------------------------------------------------------------------------------------------------

# DEFINE FUNCTION TO SOLVE FOR STEADY STATES IN PARALLEL
def solve_steady_state(rownum, ParamPolygon):
    
    alpha_val = ParamPolygon[rownum][0]
    n_val = ParamPolygon[rownum][1]
    
    xss = ssfinder(alpha_val,n_val)
    return xss, rownum

# DEFINE FUNCTION TO COMPUTE SENSITIVITIES IN PARALLEL
def compute_sensitivities(rownum, ParamPolygon, xssPolygon):
    alpha_val = ParamPolygon[rownum, 0]
    n_val = ParamPolygon[rownum, 1]

    xss_val = xssPolygon[rownum]

    return np.array([
        S_alpha_xss_analytic(xss_val, alpha_val, n_val),
        S_n_xss_analytic(xss_val, alpha_val, n_val)])

# ---------------------------------------------------------------------------------------------------

# DEFINE FUNCTION THAT RETURNS A 2D NUMPY ARRAY CONTAINING ROWS FROM ARR2 THAT ARE NOT IN ARR1, WHERE ARR1 AND ARR2 ARE 2D NUMPY ARRAYS
def rows_not_in_arr1(arr1, arr2):
    dtype = [('f0', arr2.dtype), ('f1', arr2.dtype)]
    arr1_struct = arr1.view(dtype)
    arr2_struct = arr2.view(dtype)
    unique_rows = np.setdiff1d(arr2_struct, arr1_struct).view(arr2.dtype).reshape(-1, arr2.shape[1])
    return unique_rows