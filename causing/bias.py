import logging

import numpy as np
import numdifftools
from scipy.optimize import minimize

log = logging.getLogger(__name__)


def sse_bias(bias, bias_ind, model, xdat, fym, selwei, ymdat):
    """sum of squared errors given modification indicator, Tikhonov not used"""
    bias = bias[0]

    yhat = model.compute(xdat, bias, bias_ind)
    ymhat = fym @ yhat
    err = ymhat - ymdat
    sse = np.sum(err * err * np.diag(selwei).reshape(-1, 1))

    log.info("sse {:10f}, bias {:10f}".format(sse, bias))
    return sse


def optimize_biases(model, xdat, fym, selwei, bias_ind, ymdat):
    """numerical optimize modification indicator for single equation"""

    # optimizations parameters
    bias_start = 0
    method = "BFGS"  # BFGS, SLSQP, Nelder-Mead, Powell, TNC, COBYLA, CG

    log.info("\nEstimation of bias for {}:".format(model.yvars[bias_ind]))
    out = minimize(
        sse_bias,
        bias_start,
        args=(bias_ind, model, xdat, fym, selwei, ymdat),
        method=method,
    )

    bias = out.x
    sse = out.fun

    if hasattr(out, "hess_inv"):
        hess_i = np.linalg.inv(out.hess_inv)
        log.info("Scalar Hessian from method {}.".format(method))
    else:
        hess_i = numdifftools.Derivative(sse_bias, n=2)(
            bias, bias_ind, model, xdat, fym, selwei, ymdat
        )
        log.info("Scalar Hessian computed numerically.")

    return bias, hess_i, sse


def estimate_biases(m, xdat, ymvars, ymdat):
    """numerical optimize modification indicators for equations, one at a time"""
    # checks
    pdim = len(ymvars)
    if ymdat.shape[0] != pdim:
        raise ValueError(
            "Number of ymvars {} and ymdat {} not identical.".format(
                ymdat.shape[0], pdim
            )
        )

    # prepare data
    ymmean = ymdat.mean(axis=1)
    ymcdat = ymdat - ymmean.reshape(pdim, 1)

    selvec = np.zeros(m.ndim)
    selvec[[list(m.yvars).index(el) for el in ymvars]] = 1
    # selwei whitening matrix of manifest demeaned variables
    selwei = np.diag(1 / np.var(ymcdat, axis=1))
    fym = np.eye(m.ndim)[selvec == 1]

    # estimate biases
    tau = xdat.shape[1]
    biases = np.zeros(m.ndim)
    biases_std = np.zeros(m.ndim)
    for bias_ind in range(m.ndim):
        # compute biases
        bias, hess_i, sse = optimize_biases(m, xdat, fym, selwei, bias_ind, ymdat)
        biases[bias_ind] = bias

        # compute biases_std
        resvar = sse / (tau - 1)
        bias_std = (2 * resvar * (1 / hess_i)) ** (1 / 2)
        biases_std[bias_ind] = bias_std

    return biases, biases_std
