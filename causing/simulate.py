from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import zeros, diag, eye, ones, array

from causing import utils


@dataclass
class SimulationParams:
    xmean_true: List[float]  # mean of exogeneous data
    sigx_theo: float  # true scalar error variance of xvars
    sigym_theo: float  # true scalar error variance of ymvars
    rho: float  # true correlation within y and within x vars
    tau: int  # nr. of simulated observations


def simulate(m, sim_params):
    """simulate exogeneous x and corresponding endogeneous y data for example equations"""

    # dimensions
    ndim = len(m.yvars)
    mdim = len(m.xvars)
    selvec = zeros(ndim)
    selvec[[list(m.yvars).index(el) for el in m.ymvars]] = 1
    selmat = diag(selvec)
    selvecc = selvec.reshape(ndim, 1)
    fym = eye(ndim)[diag(selmat) == 1]

    # compute theoretical RAM covariance matrices # ToDo: needed? # yyy
    sigmax_theo = sim_params.sigx_theo * (
        sim_params.rho * ones((mdim, mdim)) + (1 - sim_params.rho) * eye(mdim)
    )
    sigmau_theo = sim_params.sigym_theo * (
        sim_params.rho * selvecc @ selvecc.T + (1 - sim_params.rho) * selmat
    )

    # symmetrize covariance matrices, such that numerically well conditioned
    sigmax_theo = (sigmax_theo + sigmax_theo.T) / 2
    sigmau_theo = (sigmau_theo + sigmau_theo.T) / 2

    # simulate x data
    # use cholesky to avoid numerical random normal posdef problem, old:
    # xdat = multivariate_normal(sim_params.xmean_true, sigmax_theo, sim_params.tau).T
    xdat = utils.multivariate_normal(zeros(mdim), eye(mdim), sim_params.tau).T
    xdat = (
        array(sim_params.xmean_true).reshape(mdim, 1)
        + utils.cholesky(sigmax_theo) @ xdat
    )

    # ymdat from yhat with enndogenous errors
    yhat = m.compute(xdat)
    ymdat = fym @ (
        yhat + utils.multivariate_normal(zeros(ndim), sigmau_theo, sim_params.tau).T
    )

    # delete nan columns
    colind = ~np.any(np.isnan(ymdat), axis=0)
    if sum(colind) > 0:
        xdat = xdat[:, colind]
        ymdat = ymdat[:, colind]

    # new tau after None columns deleted
    tau_new = ymdat.shape[1]
    if tau_new < sim_params.tau:
        raise ValueError(
            "Model observations reduced from {} to {} because some simulations failed.".format(
                sim_params.tau, tau_new
            )
        )

    # test bias estimation
    # ymdat[-1, :] += 66

    return xdat, ymdat
