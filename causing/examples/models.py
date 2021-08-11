# -*- coding: utf-8 -*-
"""Model Examples."""

from sympy import symbols
from causing import utils
from causing.simulate import SimulationParams, simulate
from causing.model import Model


def example():
    """model example"""

    X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])
    equations = (
        X1,
        X2 + 2 * Y1 ** 2,
        Y1 + Y2,
    )

    model_dat = {
        "equations": equations,  # equations in topological order
        "xvars": [X1, X2],  # exogenous variables in desired order
        "yvars": [Y1, Y2, Y3],  # endogenous variables in topological order
        "ymvars": [Y3],  # manifest endogenous variables
        "final_var": Y3,  # final variable of interest, for mediation analysis
        "show_nr_indiv": 3,  # show first individual effects
    }

    # simulate data
    m = Model(
        model_dat["xvars"],
        model_dat["yvars"],
        model_dat["ymvars"],
        equations=equations,
        final_var=model_dat["final_var"],
    )
    xdat, ymdat = simulate(
        m,
        SimulationParams(
            xmean_true=[3, 2],  # mean of exogeneous data
            sigx_theo=1,  # true scalar error variance of xvars
            sigym_theo=1,  # true scalar error variance of ymvars
            rho=0.2,  # true correlation within y and within x vars
            tau=200,  # nr. of simulated observations
        ),
    )
    model_dat["xdat"] = xdat  # exogenous data

    estimate_input = dict(
        ymdat=ymdat,
        estimate_bias=True,  # estimate equation biases, for model validation
        alpha=None,  # regularization parameter, is estimated if None
        dof=None,  # effective degrees of freedom, corresponding to alpha
    )

    return model_dat, estimate_input


def example2():
    """model example 2, no regularization required, no latent variables"""

    X1, Y1 = symbols(
        [
            "X1",
            "Y1",
        ]
    )
    equations = (X1,)

    model_dat = {
        "equations": equations,
        "xvars": [X1],
        "yvars": [Y1],
        "ymvars": [Y1],
        "final_var": Y1,
        "show_nr_indiv": 3,
    }

    # simulate data
    m = Model(
        model_dat["xvars"],
        model_dat["yvars"],
        model_dat["ymvars"],
        equations=equations,
        final_var=model_dat["final_var"],
    )
    xdat, ymdat = simulate(
        m,
        SimulationParams(
            xmean_true=[3],
            sigx_theo=1,
            sigym_theo=1,
            rho=0.2,
            tau=200,
        ),
    )
    model_dat["xdat"] = xdat

    estimate_input = dict(
        ymdat=ymdat,
        estimate_bias=True,
        alpha=None,
        dof=None,
    )
    return model_dat, estimate_input


def example3():
    """model example 3

    difficult to estimate:
    if just Y3 is manifest, huge regularization is required and direct effects are strongly biased,
    (if all yvars are manifest, just slight regularization is required and some standard errors are huge)
    """

    X1, Y1, Y2, Y3 = symbols(["X1", "Y1", "Y2", "Y3"])
    equations = (
        2 * X1,
        -X1,
        Y1 + Y2,
    )

    model_dat = {
        "equations": equations,
        "xvars": [X1],
        "yvars": [Y1, Y2, Y3],
        "ymvars": [Y3],
        "final_var": Y3,
        "show_nr_indiv": 3,
    }

    # simulate data
    m = Model(
        model_dat["xvars"],
        model_dat["yvars"],
        model_dat["ymvars"],
        equations=equations,
        final_var=model_dat["final_var"],
    )
    xdat, ymdat = simulate(
        m,
        SimulationParams(
            xmean_true=[3],
            sigx_theo=1,
            sigym_theo=1,
            rho=0.2,
            tau=200,
        ),
    )
    model_dat["xdat"] = xdat

    estimate_input = dict(
        ymdat=ymdat,
        estimate_bias=True,
        alpha=None,
        dof=None,
    )

    return model_dat, estimate_input


def education():
    """Education

    The dataset contains following variables in this order, the variables 0. to 4. being time varying and variables 5. to 9. being time invariant:

    0. PERSONID = Person id (ranging from 1 to 2,178) # not used by us
    1. EDUC     = Education (years of schooling)
    2. LOGWAGE  = Log of hourly wage, at most recent job, in real 1993 dollars # we use wage instead of log wage
    3. POTEXPER = Potential experience (= AGE - EDUC - 5)
    4. TIMETRND = Time trend (starting at 1 in 1979 and incrementing by year) # not used by us
    5. ABILITY  = Ability (cognitive ability measured by test score)
    6. MOTHERED = Mother's education (highest grade completed, in years)
    7. FATHERED = Father's education (highest grade completed, in years)
    8. BRKNHOME = Dummy variable for residence in a broken home at age 14
    9. SIBLINGS = Number of siblings

    Model identified without regularization if wage instead of logwage and all observations. # yyyy

    ToDo: Automatic Hessian gives wrong results for this example: # yyyy
    Algebraic and numeric   Hessian allclose: True.
    Automatic and numeric   Hessian allclose: False.
    Automatic and algebraic Hessian allclose: False.
    No problem if ABILITY has zero effect
    """

    (
        FATHERED,
        MOTHERED,
        SIBLINGS,
        BRKNHOME,
        ABILITY,
        AGE,
        EDUC,
        POTEXPER,
        WAGE,
    ) = symbols(
        [
            "FATHERED",
            "MOTHERED",
            "SIBLINGS",
            "BRKNHOME",
            "ABILITY",
            "AGE",
            "EDUC",
            "POTEXPER",
            "WAGE",
        ]
    )

    # note that in Sympy some operators are special, e.g. Max() instead of max()
    from sympy import Max

    equations = (
        # EDUC
        13
        + 0.1 * (FATHERED - 12)
        + 0.1 * (MOTHERED - 12)
        - 0.1 * SIBLINGS
        - 0.5 * BRKNHOME,
        # POTEXPER
        Max(AGE - EDUC - 5, 0),
        # WAGE
        7 + 1 * (EDUC - 12) + 0.5 * POTEXPER + 1 * ABILITY,
    )

    model_dat = {
        "equations": equations,
        "xvars": [FATHERED, MOTHERED, SIBLINGS, BRKNHOME, ABILITY, AGE],
        "yvars": [EDUC, POTEXPER, WAGE],
        "ymvars": [EDUC, POTEXPER, WAGE],
        "final_var": WAGE,
        "show_nr_indiv": 3,
    }

    # load and transform data
    from numpy import array, concatenate, exp, loadtxt

    xymdat = loadtxt("data/education.csv", delimiter=",").reshape(-1, 10)
    xymdat = xymdat.T  # observations in columns
    # xymdat = xymdat[:, 0:200]      # just some of the 17,919 observations
    xdat = xymdat[[7, 6, 9, 8, 5]]  # without PERSONID, TIMETRND
    age = array(xymdat[3, :] + xymdat[1, :] + 5).reshape(
        1, -1
    )  # age = POTEXPER + EDUC + 5
    ymdat = xymdat[[1, 3, 2]]
    ymdat[2, :] = exp(ymdat[2, :])  # wage instead of log wage
    xdat = concatenate((xdat, age))
    model_dat["xdat"] = xdat

    estimate_input = dict(
        ymdat=ymdat,
        estimate_bias=True,
        alpha=2.637086,
        dof=0.068187,
    )

    return model_dat, estimate_input
