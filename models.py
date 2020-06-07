# -*- coding: utf-8 -*-
"""Model Examples."""

from sympy import symbols


def example():
    """model example"""

    X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])

    def define_equations(X1, X2):

        eq_Y1 = X1
        eq_Y2 = X2 + 2 * Y1**2
        eq_Y3 = Y1 + Y2

        return eq_Y1, eq_Y2, eq_Y3

    model_dat = {
        "define_equations": define_equations,   # equations in topological order
        "xvars": [X1, X2],                      # exogenous variables in desired order
        "yvars": [Y1, Y2, Y3],                  # endogenous variables in topological order
        "ymvars": [Y3],                         # manifest endogenous variables
        "final_var": Y3,                        # final variable of interest, for mediation analysis
        "show_nr_indiv": 3,                     # show first individual effects
        "alpha": None,                          # regularization parameter, is estimated if None
        "dir_path": "output/",                  # output directory path
        }

    # simulate data
    import utils
    simulation_dat = {
        "xmean_true": [3, 2],                   # mean of exogeneous data
        "sigx_theo": 1,                         # true scalar error variance of xvars
        "sigym_theo": 1,                        # true scalar error variance of ymvars
        "rho": 0.2,                             # true correlation within y and within x vars
        "tau": 200,                             # nr. of simulated observations
        }
    model_dat.update(simulation_dat)
    xdat, ymdat = utils.simulate(model_dat)

    # save data
# =============================================================================
#     from numpy import savetxt
#     savetxt("data/xdat.csv", xdat, delimiter=",")
#     savetxt("data/ymdat.csv", ymdat, delimiter=",")
# =============================================================================

    # load data
# =============================================================================
#     from numpy import loadtxt
#     xdat = loadtxt("data/xdat.csv", delimiter=",").reshape(len(model_dat["xvars"]), -1)
#     ymdat = loadtxt("data/ymdat.csv", delimiter=",").reshape(len(model_dat["ymvars"]), -1)
# =============================================================================

    model_dat["xdat"] = xdat                    # exogenous data
    model_dat["ymdat"] = ymdat                  # manifest endogenous data

    return model_dat

def example2():
    """model example 2, no regularization required, no latent variables"""

    X1, Y1 = symbols(["X1", "Y1",])

    def define_equations(X1):

        eq_Y1 = X1

        return [eq_Y1]

    model_dat = {
        "define_equations": define_equations,
        "xvars": [X1],
        "yvars": [Y1],
        "ymvars": [Y1],
        "final_var": Y1,
        "show_nr_indiv": 3,
        "alpha": None,
        "dir_path": "output/",
        }

    # simulate data
    import utils
    simulation_dat = {
        "xmean_true": [3],
        "sigx_theo": 1,
        "sigym_theo": 1,
        "rho": 0.2,
        "tau": 200,
        }
    model_dat.update(simulation_dat)
    xdat, ymdat = utils.simulate(model_dat)

    model_dat["xdat"] = xdat
    model_dat["ymdat"] = ymdat

    return model_dat

def example3():
    """model example 3
    
    difficult to estimate:
    if just Y3 is manifest, huge regularization is required and direct effects are strongly biased,
    (if all yvars are manifest, just slight regularization is required and some standard errors are huge)
    """

    X1, Y1, Y2, Y3 = symbols(["X1", "Y1", "Y2", "Y3"])

    def define_equations(X1):

        eq_Y1 = 2 * X1
        eq_Y2 = -X1
        eq_Y3 = Y1 + Y2

        return eq_Y1, eq_Y2, eq_Y3

    model_dat = {
        "define_equations": define_equations,
        "xvars": [X1],
        "yvars": [Y1, Y2, Y3],
        "ymvars": [Y3],
        "final_var": Y3,
        "show_nr_indiv": 3,
        "alpha": None,
        "dir_path": "output/",
        }

    # simulate data
    import utils
    simulation_dat = {
        "xmean_true": [3],
        "sigx_theo": 1,
        "sigym_theo": 1,
        "rho": 0.2,
        "tau": 200,
        }
    model_dat.update(simulation_dat)
    xdat, ymdat = utils.simulate(model_dat)

    model_dat["xdat"] = xdat
    model_dat["ymdat"] = ymdat

    return model_dat

def education():
    """Education"""

    (FATHERED, MOTHERED, SIBLINGS, BRKNHOME, ABILITY, AGE, EDUC, POTEXPER, LOGWAGE) = symbols(
        ["FATHERED", "MOTHERED", "SIBLINGS", "BRKNHOME", "ABILITY", "AGE", "EDUC", "POTEXPER", "LOGWAGE"])
    
    # note that in Sympy some operators are special, e.g. Max() instead of max()
    from sympy import Max

    def define_equations(FATHERED, MOTHERED, SIBLINGS, BRKNHOME, ABILITY, AGE):
        
        eq_EDUC = 12 + 0.1 * (FATHERED - 12) + 0.1 * (MOTHERED - 12) - 0.1 * SIBLINGS - 0.1 * BRKNHOME
        eq_POTEXPER = Max(AGE - EDUC - 5, 0)
        eq_LOGWAGE = 1.5 + 0.1 * (EDUC - 12) + 0.1 * POTEXPER + 0.1 * ABILITY

        return eq_EDUC, eq_POTEXPER, eq_LOGWAGE

    model_dat = {
        "define_equations": define_equations,
        "xvars": [FATHERED, MOTHERED, SIBLINGS, BRKNHOME, ABILITY, AGE],
        "yvars": [EDUC, POTEXPER, LOGWAGE],
        "ymvars": [EDUC, POTEXPER, LOGWAGE],
        "final_var": LOGWAGE,
        "show_nr_indiv": 3,
        "alpha": 16476.332322,
        "dir_path": "output/",
        }

    # load data
    from numpy import array, concatenate, loadtxt
    xymdat = loadtxt("data/education.csv", delimiter=",").reshape(-1, 10)
    xymdat = xymdat.T # observations in columns
    xymdat = xymdat[:, 0:200] # just some of the 17,919 observations # yyyy
    xdat = xymdat[[7, 6, 9, 8, 5]] # without PERSONID, TIMETRND
    age = array(xymdat[3, :] + xymdat[1, :] + 5).reshape(1, -1) # age = POTEXPER + EDUC + 5
    ymdat = xymdat[[1, 3, 2]]
    xdat = concatenate((xdat, age))
    
    model_dat["xdat"] = xdat
    model_dat["ymdat"] = ymdat

    return model_dat
