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
    """Education
    
    Gary Koop and Justin L. Tobias, "Learning about Heterogeneity in Returns
    to Schooling", Journal of Applied Econometrics, Vol. 19, No. 7, 2004,
    pp. 827-849.
    https://www.economics.uci.edu/files/docs/workingpapers/2001-02/Tobias-07.pdf

    This panel data set consists of NT=17,919 observations from  N=2,178
    individuals. It contains the wage earnings history for young workers in
    the U.S. from 1979 until 1993. The data are taken from the National
    Longitudinal Survey of Youth (NLSY).
    http://people.stern.nyu.edu/wgreene/Econometrics/PanelDataSets.htm
    
    The NLSY is a rich panel study of 12,686 individuals in the U.S. ranging in
    age from 14-22 as of the first interview date in 1979. It contains detailed
    information on the earnings and wages, educational attainment, family
    characteristics, and test scores of the sampled individuals.
    
    Koop and Tobias (2004) use a version of the NLSY which allows to obtain an
    earnings history until 1993. To abstract from selection issues in
    employment, and to remain consistent with the majority of the literature,
    they focus on the outcomes of white males in the NLSY. They restrict
    attention to those individuals who are active in the labor force for a good
    portion of each year, being at least 16 years of age in the given year, who
    reported working at least 30 weeks a year and at least 800 hours per year.
    They also deleted observations when the reported hourly wage is less than
    $1 or greater than $100 dollars per hour, when education decreases across
    time for an individual, or when the reported change in years of schooling
    over time is not consistent with the change in time from consecutive
    interviews. As such, they are careful to delete individuals whose education
    is clearly mis-measured.   
    
    The dataset contains following variables in this order:
    
    Time Varying
    0) PERSONID = Person id (ranging from 1 to 2,178) # not used by us
    1) EDUC = Education (years of schooling)
    2) LOGWAGE = Log of hourly wage, at most recent job, in real 1993 dollars
    3) POTEXPER = Potential experience (= age - EDUC- 5)
    4) TIMETRND = Time trend (starting at 1 in 1979 and incrementing by year)
    
    Time Invariant
    5) ABILITY = Ability (cognitive ability measured by test score*)
    6) MOTHERED = Mother's education (highest grade completed, in years)
    7) FATHERED = Father's education (highest grade completed, in years)
    8) BRKNHOME = Dummy variable for residence in a broken home at age 14
    9) SIBLINGS = Number of siblings
    
    * constructed from the 10 component tests of the Armed Services Vocational
    Aptitude Battery (ASVAB) administered to the NLSY participants in 1980.
    Since individuals varied in age, each of the 10 tests is first residualized
    on age, and the standardized test score is defined as the first principal
    component of the standardized residuals.
    """

    (TIMETRND, FATHERED, MOTHERED, SIBLINGS, BRKNHOME,
     EDUC, POTEXPER, ABILITY, LOGWAGE) = symbols(
         ["TIMETRND", "FATHERED", "MOTHERED", "SIBLINGS", "BRKNHOME",
          "EDUC", "POTEXPER", "ABILITY", "LOGWAGE"])

    def define_equations(TIMETRND, FATHERED, MOTHERED, SIBLINGS, BRKNHOME):
        
        eq_EDUC = 0.4 * TIMETRND + 1 * FATHERED + 1 * MOTHERED - 1 * SIBLINGS - 1 * BRKNHOME
        eq_POTEXPER = 0.6 * TIMETRND - 1 * EDUC
        eq_ABILITY = 1 * EDUC + 1 * FATHERED + 1 * MOTHERED + 1 * SIBLINGS        
        eq_LOGWAGE = 1 * EDUC + 1 * POTEXPER

        return eq_EDUC, eq_POTEXPER, eq_ABILITY, eq_LOGWAGE

    model_dat = {
        "define_equations": define_equations,
        "xvars": [TIMETRND, FATHERED, MOTHERED, SIBLINGS, BRKNHOME],
        "yvars": [EDUC, POTEXPER, ABILITY, LOGWAGE],
        "ymvars": [EDUC, POTEXPER, ABILITY, LOGWAGE],
        "final_var": LOGWAGE,
        "show_nr_indiv": 3,
        "dir_path": "output/",
        }

    # load data
    from numpy import loadtxt
    xymdat = loadtxt("data/education.csv", delimiter=",").reshape(-1, 10)
    xymdat = xymdat.T # observation in columns
    xymdat = xymdat[:, 0:200] # just some of the 17,919 observations # yyyy
    xdat = xymdat[[4, 7, 6, 9, 8]] # w/o PERSONID
    ymdat = xymdat[[1, 3, 5, 2]]
    
    print(xdat)
    print(xdat.shape)
    print(ymdat)
    print(ymdat.shape)

    model_dat["xdat"] = xdat                    # exogenous data
    model_dat["ymdat"] = ymdat                  # manifest endogenous data

    return model_dat

