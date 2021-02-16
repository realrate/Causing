# -*- coding: utf-8 -*-
"""Utilities."""

# pylint: disable=invalid-name # spyder cannot read good-names from .pylintrc
# pylint: disable=E1101 # "torch has nor 'DoubleTensor' menber"

from copy import copy, deepcopy

import pydot
import sys

import numpy as np
from numpy.random import multivariate_normal, seed
from numpy import (
    allclose, array, concatenate, count_nonzero, diag, eye, empty,
    fill_diagonal, hstack, isnan, kron, median, nan, ones, reshape, std, tile,
    var, vstack, zeros)
import numdifftools as nd
from numpy.linalg import cholesky, inv, norm
from pandas import DataFrame
from scipy.optimize import minimize
from sympy import diff, Heaviside, lambdify
import torch

from causing import svg

# set numpy random seed
seed(1002)


def adjacency(model_dat):
    """numeric function for model and direct effects, identification matrics"""

    define_equations = model_dat["define_equations"]
    xvars = model_dat["xvars"]
    yvars = model_dat["yvars"]

    ndim = len(yvars)
    mdim = len(xvars)

    def equations_alg(xvars, bias=0, bias_ind=0):
        """algebraic equations plus bias containing xvars by substituting yvars"""

        equationsx = list(define_equations(*xvars))
        equationsx[bias_ind] = bias + equationsx[bias_ind]
        for bias_ind in range(ndim):
            for j in range(bias_ind + 1, ndim):
                if hasattr(equationsx[j], 'subs'):
                    equationsx[j] = equationsx[j].subs(yvars[bias_ind], equationsx[bias_ind])

        return equationsx

    # algebraic equations containing xvars and yvars
    equations = define_equations(*xvars)

    # ToDo modules do not work, therefore replace_heaviside required # yyy
    # modules = [{'Heaviside': lambda x: np.heaviside(x, 0)}, 'sympy', 'numpy']
    # modules = [{'Heaviside': lambda x: 1 if x > 0 else 0}, 'sympy', 'numpy']
    modules = ['sympy', 'numpy']

    def model(xvals, bias=0, bias_ind=0):
        """numeric model plus bias in terms of xvars"""

        # float for conversion of numpy array from scipy minimize
        equationsx = equations_alg(xvars, float(bias), bias_ind)
        model_lam = lambdify(xvars, equationsx, modules=modules)
        xvals = array(xvals).reshape(mdim, -1)
        try:
            yhat = array([model_lam(*xval) for xval in xvals.T]).T
        except Exception as e:
            # find warnings
            print(e, "\nFinding erroneous element yhat_it ...")
            for t, xval in enumerate(xvals.T):
                for i, eq in enumerate(equationsx):
                    yhat_it = eq.subs(dict(zip(xvars, xval)))
                    print(DataFrame(xval, xvars, [t]))
                    print("i = {}, t = {}, yhat_it = {} {}"
                          .format(i, t, yhat_it, type(yhat_it)))
                    print(yvars[i], "=", eq)
            raise ValueError(e)

        return yhat.astype(np.float64)

    # algebraic direct effects containing xvars and yvars
    mx_alg = array([[diff(eq, xvar) for xvar in xvars] for eq in equations])
    my_alg = array([[diff(eq, yvar) for yvar in yvars] for eq in equations])

    # algebraic direct effects as lamba function of xvars, yvars
    #   and then only as function of xvars
    mx_lamxy = lambdify((xvars, yvars), mx_alg, modules=modules)
    my_lamxy = lambdify((xvars, yvars), my_alg, modules=modules)

    def mx_lam(xvars):
        return mx_lamxy(xvars, equations_alg(xvars))

    def my_lam(xvars):
        return my_lamxy(xvars, equations_alg(xvars))

    # identification matrics for direct effects
    idx = digital(mx_alg)
    idy = digital(my_alg)

    adjacency_dat = {
        "model": model,
        "mx_alg": mx_alg,
        "my_alg": my_alg,
        "mx_lam": mx_lam,
        "my_lam": my_lam,
        "idx": idx,
        "idy": idy,
    }
    model_dat.update(adjacency_dat)

    return model_dat


def simulate(model_dat):
    """simulate exogeneous x and corresponding endogeneous y data for example equations"""

    # dimensions
    ndim = len(model_dat["yvars"])
    mdim = len(model_dat["xvars"])
    selvec = zeros(ndim)
    selvec[[list(model_dat["yvars"]).index(el) for el in model_dat["ymvars"]]] = 1
    selmat = diag(selvec)
    selvecc = selvec.reshape(ndim, 1)
    fym = eye(ndim)[diag(selmat) == 1]

    # compute theoretical RAM covariance matrices # ToDo: needed? # yyy
    sigmax_theo = model_dat["sigx_theo"] * (
            model_dat["rho"] * ones((mdim, mdim)) + (1 - model_dat["rho"]) * eye(mdim))
    sigmau_theo = model_dat["sigym_theo"] * (
            model_dat["rho"] * selvecc @ selvecc.T + (1 - model_dat["rho"]) * selmat)

    # symmetrize covariance matrices, such that numerically well conditioned
    sigmax_theo = (sigmax_theo + sigmax_theo.T) / 2
    sigmau_theo = (sigmau_theo + sigmau_theo.T) / 2

    # simulate x data
    # use cholesky to avoid numerical random normal posdef problem, old:
    # xdat = multivariate_normal(model_dat["xmean_true"], sigmax_theo, model_dat["tau"]).T
    xdat = multivariate_normal(zeros(mdim), eye(mdim), model_dat["tau"]).T
    xdat = array(model_dat["xmean_true"]).reshape(mdim, 1) + cholesky(sigmax_theo) @ xdat

    # ymdat from yhat with enndogenous errors
    model = adjacency(model_dat)["model"]  # model constructed from adjacency
    yhat = model(xdat)
    ymdat = fym @ (yhat + multivariate_normal(zeros(ndim), sigmau_theo, model_dat["tau"]).T)

    # delete nan columns
    colind = ~np.any(isnan(ymdat), axis=0)
    if sum(colind) > 0:
        xdat = xdat[:, colind]
        ymdat = ymdat[:, colind]

    # new tau after None columns deleted
    tau_new = ymdat.shape[1]
    if tau_new < model_dat["tau"]:
        raise ValueError("Model observations reduced from {} to {} because some simulations failed."
                         .format(model_dat["tau"], tau_new))

    # test bias estimation
    # ymdat[-1, :] += 66

    return xdat, ymdat


def replace_heaviside(mxy, xvars, xval):
    """deal with sympy Min and Max giving Heaviside:
    Heaviside(x) = 0 if x < 0 and 1 if x > 0, but
    Heaviside(0) needs to be defined by user,
    we set Heaviside(0) to 0 because in general there is no sensititvity,
    the numpy heaviside function is lowercase and wants two arguments:
    an x value, and an x2 to decide what should happen for x==0
    https://stackoverflow.com/questions/60171926/sympy-name-heaviside-not-defined-within-lambdifygenerated
    """

    for i in range(mxy.shape[0]):
        for j in range(mxy.shape[1]):
            if hasattr(mxy[i, j], 'subs'):
                # ToDo: rename, check # yyyy
                # just for german_insurance substitute xvars again since
                # mxy still has sympy xvars reintroduced via yvars_elim
                mxy[i, j] = mxy[i, j].subs(dict(zip(xvars, xval)))

                # if mxy[i, j] != mxy[i, j].subs(Heaviside(0), 0):
                #    print("replaced {} by {} in element {} {}"
                #          .format(mxy[i, j], mxy[i, j].subs(Heaviside(0), 0), i, j))
                mxy[i, j] = mxy[i, j].subs(Heaviside(0), 0)

    return mxy.astype(np.float64)


def create_model(model_dat):
    """specify model and compute effects"""

    # dimensions
    ndim = len(model_dat["yvars"])
    mdim = len(model_dat["xvars"])
    pdim = len(model_dat["ymvars"])
    selvec = zeros(ndim)
    selvec[[list(model_dat["yvars"]).index(el) for el in model_dat["ymvars"]]] = 1
    selmat = diag(selvec)
    tau = model_dat["xdat"].shape[1]
    selvec = diag(selmat)

    # check
    if model_dat["ymdat"].shape[0] != pdim:
        raise ValueError("Number of yvars {} and ymdat {} not identical."
                         .format(model_dat["ymdat"].shape[0], pdim))

    # numeric function for model and direct effects, identification matrics
    model_dat.update(adjacency(model_dat))

    # yhat without enndogenous errors
    yhat = model_dat["model"](model_dat["xdat"])
    yhat = vstack(yhat).reshape(len(model_dat["yvars"]), -1)

    # means and demeaning data for estimation of linear total derivative
    xmean = model_dat["xdat"].mean(axis=1)
    ydet = model_dat["model"](xmean)
    ymean = yhat.mean(axis=1)
    ymmean = model_dat["ymdat"].mean(axis=1)
    ymedian = median(yhat, axis=1)
    xmedian = median(model_dat["xdat"], axis=1)
    xcdat = model_dat["xdat"] - xmean.reshape(mdim, 1)
    ymcdat = model_dat["ymdat"] - ymmean.reshape(pdim, 1)

    # effect identification matrices
    edx, edy = compute_ed(model_dat["idx"], model_dat["idy"])
    _, _, fdx, fdy = compute_fd(model_dat["idx"], model_dat["idy"],
                                model_dat["yvars"], model_dat["final_var"])

    # more dimensions
    qxdim = count_nonzero(model_dat["idx"])
    qydim = count_nonzero(model_dat["idy"])
    qdim = qxdim + qydim

    # model summary
    print("Causing starting")
    print("\nModel with {} endogenous and {} exogenous variables, "
          "{} direct effects and {} observations."
          .format(ndim, mdim, qdim, tau))

    # individual theoretical effects
    (mx_theos, my_theos, ex_theos, ey_theos, exj_theos, eyx_theos, eyj_theos, eyy_theos
     ) = ([] for i in range(8))
    for obs in range(min(tau,
                         model_dat["show_nr_indiv"])):
        xval = model_dat["xdat"][:, obs]

        # numeric direct effects since no sympy algebraic derivative
        mx_theo = replace_heaviside(array(model_dat["mx_lam"](xval)), model_dat["xvars"], xval)  # yyy
        my_theo = replace_heaviside(array(model_dat["my_lam"](xval)), model_dat["xvars"], xval)  # yyy

        # total and final effects
        ex_theo, ey_theo = total_effects_alg(mx_theo, my_theo, edx, edy)
        exj_theo, eyj_theo, eyx_theo, eyy_theo = compute_mediation_effects(
            mx_theo, my_theo, ex_theo, ey_theo, model_dat["yvars"], model_dat["final_var"])

        # append
        mx_theos.append(mx_theo)
        my_theos.append(my_theo)
        ex_theos.append(ex_theo)
        ey_theos.append(ey_theo)
        exj_theos.append(exj_theo)
        eyx_theos.append(eyx_theo)
        eyj_theos.append(eyj_theo)
        eyy_theos.append(eyy_theo)

    # theoretical total effects at xmean and corresponding consistent ydet,
    # using closed form algebraic formula from sympy direct effects
    #   instead of automatic differentiation of model
    mx_theo = replace_heaviside(array(model_dat["mx_lam"](xmean)), model_dat["xvars"], xmean)  # yyy
    my_theo = replace_heaviside(array(model_dat["my_lam"](xmean)), model_dat["xvars"], xmean)  # yyy

    ex_theo, ey_theo = total_effects_alg(mx_theo, my_theo, edx, edy)
    exj_theo, eyj_theo, eyx_theo, eyy_theo = compute_mediation_effects(
        mx_theo, my_theo, ex_theo, ey_theo, model_dat["yvars"], model_dat["final_var"])
    direct_theo = directvec_alg(mx_theo, my_theo, model_dat["idx"], model_dat["idy"])

    # selwei whitening matrix of manifest demeaned variables
    selwei = diag(1 / var(ymcdat, axis=1))

    # ToDo: some entries are just passed directly, use update for others
    setup_dat = {
        "direct_theo": direct_theo,
        "selmat": selmat,
        "selvec": selvec,
        "selwei": selwei,
        "ymean": ymean,
        "xmedian": xmedian,
        "ymedian": ymedian,
        "ydet": ydet,
        "tau": tau,
        "xcdat": xcdat,
        "ymcdat": ymcdat,
        "yhat": yhat,
        "xmean": xmean,
        "mx_theo": mx_theo,
        "my_theo": my_theo,
        "ex_theo": ex_theo,
        "ey_theo": ey_theo,
        "exj_theo": exj_theo,
        "eyx_theo": eyx_theo,
        "eyj_theo": eyj_theo,
        "eyy_theo": eyy_theo,
        "mx_theos": mx_theos,
        "my_theos": my_theos,
        "ex_theos": ex_theos,
        "ey_theos": ey_theos,
        "exj_theos": exj_theos,
        "eyx_theos": eyx_theos,
        "eyj_theos": eyj_theos,
        "eyy_theos": eyy_theos,
    }

    model_dat.update(setup_dat)
    model_dat = update_model(model_dat)

    return model_dat


def nonzero(el):
    """identifies nonzero element"""

    if el == 0:
        nonz = 0
    if el != 0:
        nonz = 1

    return nonz


def roundec(num, dec=None):
    """rounds number or string to dec decimals,
    converts to string and strips trailing zeros and dot from the right"""

    # dec
    if not dec:
        limit_dec = 1000  # ToDo: set limit_dec globally # yyy
        if abs(num) < limit_dec:
            dec = 2
        else:
            dec = 0

    string = ("{0:." + str(dec) + "f}").format(float(num)).rstrip("0").rstrip(".")

    return string


def submatrix(mat, j):
    """computes submatrix or -vector by replacing j-th row and column by zeros"""

    ndim = mat.shape[0]
    mdim = mat.shape[1]
    sub = deepcopy(mat)

    mzeros = zeros(mdim)
    nzeros = zeros(ndim)

    if ndim > 1:
        sub[j, :] = mzeros
    if mdim > 1:
        sub[:, j] = nzeros

    return sub


def compute_ed(idx, idy):
    """compute total effects identification matrices
    from direct identification matrices or direct effects"""

    edx, edy = total_effects_alg(idx, idy, None, None)

    edx = digital(edx)
    edy = digital(edy)

    return edx, edy


def compute_fd(idx, idy, yvars, final_var):
    """compute mediation effects identification matrices
    from direct identification matrices or direct effectss"""

    edx, edy = compute_ed(idx, idy)
    exj, eyj, eyx, eyy = compute_mediation_effects(idx, idy, edx, edy, yvars, final_var)

    fdxj = digital(exj)
    fdyj = digital(eyj)
    fdx = digital(eyx)
    fdy = digital(eyy)

    return fdxj, fdyj, fdx, fdy


def total_effects_alg(mx, my, edx, edy):
    """compute algebraic total effects given direct effects and identification matrices"""

    # dimensions
    ndim = mx.shape[0]

    # error if my is not normalized
    if sum(abs(diag(my))) > 0:
        raise ValueError("No Normalization. Diagonal elements of 'my' differ from zero.")

    # total effects
    ey = inv(eye(ndim) - my)
    ex = ey @ mx

    # set fixed null and unity effects numerically exactly to 0 and 1
    if edx is not None:
        ex[edx == 0] = 0
    if edy is not None:
        ey[edy == 0] = 0
        fill_diagonal(ey, 1)

    return ex, ey


def sse_orig(mx, my, fym, ychat, ymcdat, selwei, model_dat):
    """weighted MSE target function plus Tikhonov regularization term"""

    # weighted mean squared error
    ymchat = fym @ ychat
    err = ymchat - ymcdat

    # sse = torch.trace(err.T @ selwei @ err) # big matrix needs too much RAM
    # elementwise multiplication and broadcasting and summation:
    sse = sum(torch.sum(err * err * torch.diag(selwei).view(-1, 1), dim=0))

    # sse with tikhonov term
    direct = directvec(mx, my, model_dat["idx"], model_dat["idy"])
    ssetikh = sse + model_dat["alpha"] * direct.T @ direct

    return ssetikh.requires_grad_(True)


class StructuralNN(torch.nn.Module):
    """AD identified structural linear nn,
    linear ychat approximation using ex effects reduced form"""

    def __init__(self, model_dat):
        super(StructuralNN, self).__init__()

        self.eye = torch.DoubleTensor(eye(model_dat["ndim"]))
        self.idx = torch.DoubleTensor(model_dat["idx"])
        self.idy = torch.DoubleTensor(model_dat["idy"])
        self.xcdat = torch.DoubleTensor(model_dat["xcdat"])

    def forward(self, mx, my):
        # impose identification restrictions already on imput
        # ToDo: use torch.nn.utils.prune custom_from_mask or own custom method
        mx = mx * self.idx
        my = my * self.idy

        ey = (self.eye - my).inverse()
        ex = ey @ mx
        dy = ex @ self.xcdat  # reduced form
        ychat = dy

        return ychat


def optimize_ssn(ad_model, mx, my, fym, ydata, selwei, model_dat,
                 optimizer, params, do_print=True):
    """ad torch optimization of structural neural network"""

    # parameters
    rel = 0.0001  # ToDo: define globally
    nr_conv_min = 5  # ToDo: define globally

    sse = torch.DoubleTensor([0])
    sse_old = torch.DoubleTensor([1])
    nr_conv = 0
    epoch = 0
    while nr_conv < nr_conv_min:
        sse_old = copy(sse)
        ychat = ad_model(*params)
        sse = sse_orig(mx, my, fym, ychat, ydata, selwei, model_dat)  # forward
        optimizer.zero_grad()
        sse.backward(create_graph=True)  # backward
        optimizer.step()
        if abs(sse - sse_old) / sse_old < rel:
            nr_conv += 1
        else:
            nr_conv = 0
        nrm = sum([torch.norm(param) for param in params]).detach().numpy()
        if do_print:
            print("epoch {:>4}, sse {:10f}, param norm {:10f}".format(epoch, sse.item(), nrm))
        epoch += 1

    return sse


def estimate_snn(model_dat, do_print=True):
    """estimate direct effects in identified structural form
    using PyTorch AD automatic differentiation

    forcasting y is done by reduced form since it is already solved for dy
    structural form:
        dy = my @ dy + mx @ dx
        mx, my is a linear network of at most ndim + mdim layers of max of max dims max(ndim, mdim)
        with identifiying restrictions idx, idy
    reduced form:
        dy = ex @ dx
        ex is a linear network with one layer of dimension (ndim, mdim)
        with restrictions edx
    Estimating effects with automatic differentiation only works for DAG
    """

    fym = torch.DoubleTensor(model_dat["fym"])
    selwei = torch.DoubleTensor(model_dat["selwei"])

    # start at theoretical direct effects
    mx = torch.DoubleTensor(deepcopy(model_dat["mx_theo"]))
    my = torch.DoubleTensor(deepcopy(model_dat["my_theo"]))

    # define optimization parameters
    ydata = torch.DoubleTensor(model_dat["ymcdat"])  # ymcdat
    mx.requires_grad_(True)
    my.requires_grad_(True)
    params = [mx, my]
    ad_model = StructuralNN(model_dat)  # ychat
    # Adam, Adadelta, Adagrad, AdamW, Adamax, RMSprop, Rprop
    optimizer = torch.optim.Rprop(params)

    if do_print:
        print("\nEstimation of direct effects using a structural neural network \n"
              "with regularization parameter alpha = {:10f}:".format(model_dat["alpha"]))
    sse = optimize_ssn(ad_model, mx, my, fym, ydata, selwei, model_dat,
                       optimizer, params, do_print)

    mx = mx.detach().numpy()
    my = my.detach().numpy()
    sse = sse.detach().numpy()
    assert allclose(mx, mx * model_dat["idx"]), \
        "idx identification restrictions not met:\n{}\n!=\n{}".format(mx, mx * model_dat["idx"])
    assert allclose(my, my * model_dat["idy"]), \
        "idy identification restrictions not met:\n{}\n!=\n{}".format(my, my * model_dat["idy"])

    return mx, my, sse


def sse_bias(bias, bias_ind, model_dat):
    """sum of squared errors given modification indicator, Tikhonov not used"""

    yhat = model_dat["model"](model_dat["xdat"], bias, bias_ind)
    ymhat = model_dat["fym"] @ yhat
    err = ymhat - model_dat["ymdat"]
    sse = np.sum(err * err * diag(model_dat["selwei"]).reshape(-1, 1))

    print("sse {:10f}, bias {:10f}".format(sse, float(bias)))

    return sse


def optimize_biases(model_dat, bias_ind):
    """numerical optimize modification indicator for single equation"""

    # optimizations parameters
    bias_start = 0
    method = 'SLSQP'  # BFGS, SLSQP, Nelder-Mead, Powell, TNC, COBYLA, CG

    print("\nEstimation of bias for {}:".format(model_dat["yvars"][bias_ind]))
    out = minimize(sse_bias, bias_start, args=(bias_ind, model_dat), method=method)

    bias = out.x
    sse = out.fun

    if hasattr(out, 'hess_inv'):
        hess_i = inv(out.hess_inv)
        print("Scalar Hessian from method {}.".format(method))
    else:
        hess_i = nd.Derivative(sse_bias, n=2)(bias, bias_ind, model_dat)
        print("Scalar Hessian numerically.")

    return bias, hess_i, sse


def sse_hess(mx, my, model_dat):
    """compute automatic Hessian of sse at given data and direct effects"""

    fym = torch.DoubleTensor(model_dat["fym"])
    ydata = torch.DoubleTensor(model_dat["ymcdat"])
    selwei = torch.DoubleTensor(model_dat["selwei"])

    def sse_orig_vec_alg(direct):
        """computes the ad target function sum of squared errors,
        input as tensor vectors, yields Hessian in usual dimension of identified parameters"""
        mx, my = directmat(direct, model_dat["idx"], model_dat["idy"])
        ad_model = StructuralNN(model_dat)
        ychat = ad_model(mx, my)
        return sse_orig(mx, my, fym, ychat, ydata, selwei, model_dat)

    direct = directvec(mx, my, model_dat["idx"], model_dat["idy"])
    hessian = torch.autograd.functional.hessian(sse_orig_vec_alg, direct)

    # symmetrize Hessian, such that numerically well conditioned
    hessian = hessian.detach().numpy()
    hessian = (hessian + hessian.T) / 2

    return hessian


def compute_mediation_effects(mx, my, ex, ey, yvars, final_var):
    """compute mediation effects for final variable

    use mediation matrix representation with final variable held fixed,
    in addition, select corresponding total effects vectors on final var"""

    # dimensions
    ndim = mx.shape[0]
    mdim = mx.shape[1]
    jvar = list(yvars).index(final_var)

    # corresponding total effects vectors on final var
    exj = ex[jvar, :]  # (mdim)
    eyj = ey[jvar, :]  # (ndim)

    # mediation effects matrices with final var held fixed
    eyx = (eyj.reshape(ndim, 1) @ ones((1, mdim))) * mx  # (ndim x mdim)
    eyy = (eyj.reshape(ndim, 1) @ ones((1, ndim))) * my  # (ndim x ndim)

    return exj, eyj, eyx, eyy


def tvals(eff, std):
    """compute t-values by element wise division of eff and std matrices"""

    assert eff.shape == std.shape

    if len(eff.shape) == 1:  # vector
        rows = eff.shape[0]
        tvalues = empty(rows) * nan
        for i in range(rows):
            if std[i] != 0:
                tvalues[i] = eff[i] / std[i]

    if len(eff.shape) == 2:  # matrix
        rows, cols = eff.shape
        tvalues = zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if std[i, j] != 0:
                    tvalues[i, j] = eff[i, j] / std[i, j]
                else:
                    tvalues[i, j] = nan

    return tvalues


def compute_mediation_std(ex_hat_std, ey_hat_std, eyx, eyy, yvars, final_var):
    """compute mediation std"""

    # dimensions
    ndim = ex_hat_std.shape[0]
    mdim = ex_hat_std.shape[1]
    jvar = list(yvars).index(final_var)

    exj_hat_std = ex_hat_std[jvar, :]  # (mdim)
    eyj_hat_std = ey_hat_std[jvar, :]  # (ndim)

    # construct matices of repeating rows
    exj_hat_std_mat = tile(exj_hat_std, (ndim, 1))  # (ndim x mdim)
    eyj_hat_std_mat = tile(eyj_hat_std, (ndim, 1))  # (ndim x ndim)

    # column sums of mediation matrices
    x_colsum = np.sum(eyx, axis=0)
    y_colsum = np.sum(eyy, axis=0)
    # normed mediation matrices by division by column sums,
    # zero sum for varaibles w/o effect on others
    # substituted by nan to avoid false interpretation
    x_colsum[x_colsum == 0] = nan
    y_colsum[y_colsum == 0] = nan
    eyx_colnorm = zeros((ndim, mdim))  # (ndim x mdim)
    eyx_colnorm[:] = nan
    for j in range(mdim):
        if not isnan(x_colsum[j]):
            eyx_colnorm[:, j] = eyx[:, j] / x_colsum[j]
    eyy_colnorm = zeros((ndim, ndim))  # (ndim x ndim)
    eyy_colnorm[:] = nan
    for j in range(ndim):
        if not isnan(y_colsum[j]):
            eyy_colnorm[:, j] = eyy[:, j] / y_colsum[j]

    # mediation std matrices
    eyx_hat_std = exj_hat_std_mat * eyx_colnorm  # (ndim x mdim)
    eyy_hat_std = eyj_hat_std_mat * eyy_colnorm  # (ndim x ndim)

    return exj_hat_std, eyj_hat_std, eyx_hat_std, eyy_hat_std


def directmat_alg(direct, idx, idy):
    """algebraic direct effect matrices column-wise
    from direct effects vector and id matrices"""

    # dimensions
    ndim = idx.shape[0]
    mdim = idx.shape[1]
    qydim = count_nonzero(idy)

    # compute direct effects matrices
    my = zeros((ndim, ndim))
    my.T[idy.T == 1] = direct[0:qydim]
    mx = zeros((ndim, mdim))
    mx.T[idx.T == 1] = direct[qydim:]

    return mx, my


def directmat(direct, idx, idy):
    """automatic direct effects matrices column-wise
    from direct effects vector and id matrices"""

    # dimensions
    ndim = idx.shape[0]
    mdim = idx.shape[1]

    # compute direct effects matrices
    my = torch.DoubleTensor(zeros((ndim, ndim)))
    mx = torch.DoubleTensor(zeros((ndim, mdim)))
    k = 0
    for i in range(ndim):
        for j in range(ndim):
            if idy[i, j] == 1:
                my[i, j] = direct[k]
                k += 1
    for i in range(ndim):
        for j in range(mdim):
            if idx[i, j] == 1:
                mx[i, j] = direct[k]
                k += 1

    return mx, my


def directvec_alg(mx, my, idx, idy):
    """algebraic direct effects vector column-wise
     from direct effects matrices and id matrices"""

    directy = my.T[idy.T == 1]
    directx = mx.T[idx.T == 1]
    direct = concatenate((directy, directx), axis=0)

    return direct


def directvec(mx, my, idx, idy):
    """automatic direct effects vector column-wise
    from direct effects matrices and id matrices"""

    # dimensions
    ndim = idx.shape[0]
    mdim = idx.shape[1]
    qydim = count_nonzero(idy)
    qxdim = count_nonzero(idx)

    # compute direct effects vector
    direct = torch.DoubleTensor(zeros(qydim + qxdim))
    k = 0
    for i in range(ndim):
        for j in range(ndim):
            if idy[i, j] == 1:
                direct[k] = my[i, j]
                k += 1
    for i in range(ndim):
        for j in range(mdim):
            if idx[i, j] == 1:
                direct[k] = mx[i, j]
                k += 1

    return direct


def total_from_direct(direct, idx, idy, edx, edy):
    """construct total effects vector from direct effects vector and id and ed matrices"""

    mx, my = directmat_alg(direct, idx, idy)
    ex, ey = total_effects_alg(mx, my, edx, edy)

    effects = directvec_alg(ex, ey, edx, edy)

    return effects


def digital(mat):
    """transform a matrix or vector to digital matrix,
    elements are equal to one if original element is unequal zero, and zero otherwise"""

    if len(mat.shape) == 1:  # vector
        rows = mat.shape[0]
        mat_digital = zeros(rows)
        for i in range(rows):
            if mat[i] != 0:
                mat_digital[i] = 1

    if len(mat.shape) == 2:  # matrix
        rows = mat.shape[0]
        cols = mat.shape[1]
        mat_digital = zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if mat[i, j] != 0:
                    mat_digital[i, j] = 1

    return mat_digital


def print_output(model_dat, estimate_dat, indiv_dat):
    """print theoretical and estimated values to output file"""
    # create directory if not exist
    import os
    if not os.path.exists(model_dat["dir_path"]):
        os.makedirs(model_dat["dir_path"])
    # print output file
    stdout = sys.stdout
    fha = open(model_dat["dir_path"] + "/output.txt", 'w')
    sys.stdout = fha

    # model variables
    yx_vars = (model_dat["yvars"], model_dat["xvars"])
    yy_vars = (model_dat["yvars"], model_dat["yvars"])
    # xyvars = concatenate((model_dat["xvars"], model_dat["yvars"]), axis=0)

    # compute dataframe strings for printing
    if model_dat["estimate_bias"]:
        biases = concatenate(
            (estimate_dat["biases"].reshape(1, -1),
             estimate_dat["biases_std"].reshape(1, -1),
             (estimate_dat["biases"] / estimate_dat["biases_std"]).reshape(1, -1)))
        biases_dfstr = DataFrame(biases, ("biases", "std", "t-values"),
                                 model_dat["yvars"]).to_string()

    mx_theo_dfstr = DataFrame(model_dat["mx_theo"], *yx_vars).to_string()
    my_theo_dfstr = DataFrame(model_dat["my_theo"], *yy_vars).to_string()
    ex_theo_dfstr = DataFrame(model_dat["ex_theo"], *yx_vars).to_string()
    ey_theo_dfstr = DataFrame(model_dat["ey_theo"], *yy_vars).to_string()
    eyx_theo_dfstr = DataFrame(model_dat["eyx_theo"], *yx_vars).to_string()
    eyy_theo_dfstr = DataFrame(model_dat["eyy_theo"], *yy_vars).to_string()

    mx_hat_dfstr = DataFrame(estimate_dat["mx_hat"], *yx_vars).to_string()
    my_hat_dfstr = DataFrame(estimate_dat["my_hat"], *yy_vars).to_string()
    ex_hat_dfstr = DataFrame(estimate_dat["ex_hat"], *yx_vars).to_string()
    ey_hat_dfstr = DataFrame(estimate_dat["ey_hat"], *yy_vars).to_string()
    eyx_hat_dfstr = DataFrame(estimate_dat["eyx_hat"], *yx_vars).to_string()
    eyy_hat_dfstr = DataFrame(estimate_dat["eyy_hat"], *yy_vars).to_string()

    idx_dfstr = DataFrame(model_dat["idx"], *yx_vars).to_string()
    idy_dfstr = DataFrame(model_dat["idy"], *yy_vars).to_string()
    edx_dfstr = DataFrame(model_dat["edx"], *yx_vars).to_string()
    edy_dfstr = DataFrame(model_dat["edy"], *yy_vars).to_string()
    fdx_dfstr = DataFrame(model_dat["fdx"], *yx_vars).to_string()
    fdy_dfstr = DataFrame(model_dat["fdy"], *yy_vars).to_string()

    mx_hat_std_dfstr = DataFrame(estimate_dat["mx_hat_std"], *yx_vars).to_string()
    my_hat_std_dfstr = DataFrame(estimate_dat["my_hat_std"], *yy_vars).to_string()
    ex_hat_std_dfstr = DataFrame(estimate_dat["ex_hat_std"], *yx_vars).to_string()
    ey_hat_std_dfstr = DataFrame(estimate_dat["ey_hat_std"], *yy_vars).to_string()
    eyx_hat_std_dfstr = DataFrame(estimate_dat["eyx_hat_std"], *yx_vars).to_string()
    eyy_hat_std_dfstr = DataFrame(estimate_dat["eyy_hat_std"], *yy_vars).to_string()

    hessian_hat_dfstr = DataFrame(estimate_dat["hessian_hat"]).to_string()

    x_stats = vstack((model_dat["xmean"].reshape(1, -1),
                      model_dat["xmedian"].reshape(1, -1),
                      std(model_dat["xdat"], axis=1).reshape(1, -1),
                      ones(model_dat["mdim"]).reshape(1, -1)))
    x_stats_dfstr = DataFrame(x_stats, ["xmean", "xmedian", "std", "manifest"],
                              model_dat["xvars"]).to_string()
    ydat_stats = vstack((model_dat["ymdat"].mean(axis=1).reshape(1, -1),
                         median(model_dat["ymdat"], axis=1).reshape(1, -1),
                         std(model_dat["ymdat"], axis=1).reshape(1, -1),
                         ones(model_dat["pdim"]).reshape(1, -1)))
    ydat_stats_dfstr = DataFrame(ydat_stats, ["ymean", "ymedian", "std", "manifest"],
                                 model_dat["ymvars"]).to_string()
    yhat_stats = vstack((model_dat["ymean"].reshape(1, -1),
                         model_dat["ymedian"].reshape(1, -1),
                         model_dat["ydet"].reshape(1, -1),
                         std(model_dat["yhat"], axis=1).reshape(1, -1),
                         diag(model_dat["selmat"]).reshape(1, -1)))
    yhat_stats_dfstr = DataFrame(yhat_stats, ["ymean", "ymedian", "ydet", "std", "manifest"],
                                 model_dat["yvars"]).to_string()

    # xydat = concatenate((model_dat["xdat"], model_dat["yhat"]), axis=0)
    # xydat_dfstr = DataFrame(xydat, xyvars, range(model_dat["tau"])).to_string()

    # dx_mat_df = DataFrame(indiv_dat["dx_mat"], model_dat["xvars"], range(model_dat["tau"]))
    # dy_mat_df = DataFrame(indiv_dat["dy_mat"], model_dat["yvars"], range(model_dat["tau"]))
    # dx_mat_dfstr = dx_mat_df.to_string()
    # dy_mat_dfstr = dy_mat_df.to_string()

    # model summary
    print("Causing output file")
    print("\nModel with {} endogenous and {} exogenous variables, "
          "{} direct effects and {} observations.".format(
        model_dat["ndim"], model_dat["mdim"], model_dat["qdim"], model_dat["tau"]))

    # alpha
    print()
    print("alpha: {:10f}, dof: {:10f}"
          .format(model_dat["alpha"], model_dat["dof"]))

    # biases
    if model_dat["estimate_bias"]:
        print()
        print("biases:")
        print(biases_dfstr)

    # algebraic direct and total effects
    print("\nmx_alg:")
    print(np.array2string(model_dat["mx_alg"]))
    print("\nmy_alg:")
    print(np.array2string(model_dat["my_alg"]))

    # descriptive statistics
    print()
    print("xdat:")
    print(x_stats_dfstr)
    print("ydat:")
    print(ydat_stats_dfstr)
    print("yhat:")
    print(yhat_stats_dfstr)

    # input and output data
    # print()
    # print("xdat, yhat:")
    # print(xydat_dfstr)

    # exogeneous direct effects
    print("\nExogeneous direct effects idx:")
    print(idx_dfstr)
    print(model_dat["idx"].shape)
    print("Exogeneous direct effects mx_theo:")
    print(mx_theo_dfstr)
    print(model_dat["mx_theo"].shape)
    print("Exogeneous direct effects mx_hat:")
    print(mx_hat_dfstr)
    print(estimate_dat["mx_hat"].shape)
    print("Exogeneous direct effects mx_hat_std:")
    print(mx_hat_std_dfstr)
    print(estimate_dat["mx_hat_std"].shape)

    # endogeneous direct effects
    print("\nEndogeneous direct effects idy:")
    print(idy_dfstr)
    print(model_dat["idy"].shape)
    print("Endogeneous direct effects my_theo:")
    print(my_theo_dfstr)
    print(model_dat["my_theo"].shape)
    print("Endogeneous direct effects my_hat:")
    print(my_hat_dfstr)
    print(estimate_dat["my_hat"].shape)
    print("Endogeneous direct effects my_hat_std:")
    print(my_hat_std_dfstr)
    print(estimate_dat["my_hat_std"].shape)

    # exogeneous total effects
    print("\nExogeneous total effects edx:")
    print(edx_dfstr)
    print(model_dat["edx"].shape)
    print("Exogeneous total effects ex_theo:")
    print(ex_theo_dfstr)
    print(model_dat["ex_theo"].shape)
    print("Exogeneous total effects ex_hat:")
    print(ex_hat_dfstr)
    print(estimate_dat["ex_hat"].shape)
    print("Exogeneous total effects ex_hat_std:")
    print(ex_hat_std_dfstr)
    print(estimate_dat["ex_hat_std"].shape)

    # endogeneous total effects
    print("\nEndogeneous total effects edy:")
    print(edy_dfstr)
    print(model_dat["edy"].shape)
    print("Endogeneous total effects ey_theo:")
    print(ey_theo_dfstr)
    print(model_dat["ey_theo"].shape)
    print("Endogeneous total effects ey_hat:")
    print(ey_hat_dfstr)
    print(estimate_dat["ey_hat"].shape)
    print("Endogeneous total effects ey_hat_std:")
    print(ey_hat_std_dfstr)
    print(estimate_dat["ey_hat_std"].shape)

    # exogeneous mediation effects
    print("\nExogeneous mediation effects fdx:")
    print(fdx_dfstr)
    print("Exogeneous mediation effects eyx_theo:")
    print(eyx_theo_dfstr)
    print("Exogeneous mediation effects eyx_hat:")
    print(eyx_hat_dfstr)
    print(estimate_dat["eyx_hat"].shape)
    print("Exogeneous mediation effects eyx_hat_std:")
    print(eyx_hat_std_dfstr)
    print(estimate_dat["eyx_hat_std"].shape)

    # endogeneous mediation effects
    print("\nEndogeneous mediation effects fdy:")
    print(fdy_dfstr)
    print("Endogeneous mediation effects eyy_theo:")
    print(eyy_theo_dfstr)
    print("Endogeneous mediation effects eyy_hat:")
    print(eyy_hat_dfstr)
    print(estimate_dat["eyy_hat"].shape)
    print("Endogeneous mediation effects eyy_hat_std:")
    print(eyy_hat_std_dfstr)
    print(estimate_dat["eyy_hat_std"].shape)

    # hessian
    print("\nAlgebraic Hessian at estimated direct effects hessian_hat:")
    print(hessian_hat_dfstr)
    print(estimate_dat["hessian_hat"].shape)

    # indiv matrices
    # print("\nExogeneous indiv matrix dx_mat:")
    # print(dx_mat_dfstr)
    # print((model_dat["mdim"], model_dat["tau"]))
    # print("\nEndogeneous indiv matrix dy_mat:")
    # print(dy_mat_dfstr)
    # print((model_dat["ndim"], model_dat["tau"]))

    # print to stdout
    sys.stdout = stdout
    fha.close()


def update_model(model_dat):
    """update all identification elements of dict model consistently"""

    # compute dict values
    ndim = model_dat["idx"].shape[0]
    mdim = model_dat["idx"].shape[1]
    pdim = len(model_dat["ymvars"])
    qxdim = count_nonzero(model_dat["idx"])
    qydim = count_nonzero(model_dat["idy"])
    qdim = qxdim + qydim
    edx, edy = compute_ed(model_dat["idx"], model_dat["idy"])
    _, _, fdx, fdy = compute_fd(model_dat["idx"], model_dat["idy"],
                                model_dat["yvars"], model_dat["final_var"])
    selvec = diag(model_dat["selmat"])
    fy = eye(ndim + mdim)[concatenate((ones(ndim), zeros(mdim))) == 1]
    fx = eye(ndim + mdim)[concatenate((zeros(ndim), ones(mdim))) == 1]
    fm = eye(ndim + mdim)[concatenate((selvec, ones(mdim))) == 1]
    fym = eye(ndim)[selvec == 1]

    # add dict keys and values
    model_dat["ndim"] = ndim
    model_dat["mdim"] = mdim
    model_dat["pdim"] = pdim
    model_dat["qxdim"] = qxdim
    model_dat["qydim"] = qydim
    model_dat["qdim"] = qdim
    model_dat["edx"] = edx
    model_dat["edy"] = edy
    model_dat["fdx"] = fdx
    model_dat["fdy"] = fdy
    model_dat["fy"] = fy
    model_dat["fx"] = fx
    model_dat["fm"] = fm
    model_dat["fym"] = fym

    return model_dat


def vecmat(mz):
    """compute matrix of individually vectorized nonzero elements of mz,
    for algebraic derivative of effects wrt. direct effects"""

    # dimensions
    ndim = mz.shape[0]
    mdim = mz.shape[1]
    qdim = count_nonzero(mz)

    vec_mat = zeros((ndim * mdim, qdim))
    k = 0
    for col in range(mdim):
        for row in range(ndim):
            if mz[row, col] != 0:
                mi = zeros((ndim, mdim))
                mi[row, col] = mz[row, col]
                vec_mat[:, k] = reshape(mi, ndim * mdim, order='F')
                k += 1

    return vec_mat


def compute_direct_std(vcm_direct_hat, model_dat):
    """compute direct effects standard deviations from direct effects covariance matrix"""

    direct_std = diag(vcm_direct_hat) ** (1 / 2)
    mx_std, my_std = directmat_alg(direct_std, model_dat["idx"], model_dat["idy"])

    return mx_std, my_std


def total_effects_std(direct_hat, vcm_direct_hat, model_dat):
    """compute total effects standard deviations

    given estimated vcm_direct_hat,
    using algebraic delta method for covariance Matrix of effects and
    algebraic gradient of total effects wrt. direct effects
    """

    # compute vec matrices for algebraic effects gradient wrt. to direct_hat
    vecmaty = vecmat(model_dat["idy"])
    vecmatx = vecmat(model_dat["idx"])
    vecmaty = hstack((vecmaty, zeros((model_dat["ndim"] * model_dat["ndim"], model_dat["qxdim"]))))
    vecmatx = hstack((zeros((model_dat["ndim"] * model_dat["mdim"], model_dat["qydim"])), vecmatx))

    # compute algebraic gradient of total effects wrt. direct effects
    mx, my = directmat_alg(direct_hat, model_dat["idx"], model_dat["idy"])
    ey = inv(eye(model_dat["ndim"]) - my)
    jac_effects_y = ((kron(ey.T, ey) - eye(model_dat["ndim"] * model_dat["ndim"]))
                     @ vecmaty + vecmaty)
    jac_effects_x = (kron((ey @ mx).T, ey) @ vecmaty
                     + kron(eye(model_dat["mdim"]), (ey - eye(model_dat["ndim"])))
                     @ vecmatx + vecmatx)

    # reduce to rows corresponding to nonzero effects
    indy = reshape(model_dat["edy"], model_dat["ndim"] * model_dat["ndim"], order='F') != 0
    indx = reshape(model_dat["edx"], model_dat["ndim"] * model_dat["mdim"], order='F') != 0
    jac_effects_y = jac_effects_y[indy, :]
    jac_effects_x = jac_effects_x[indx, :]
    jac_effects = vstack((jac_effects_y, jac_effects_x))

    # compare numeric and algebraic gradient of effects
    do_compare = True
    if do_compare:
        jac_effects_num = nd.Jacobian(total_from_direct)(
            direct_hat, model_dat["idx"], model_dat["idy"], model_dat["edx"], model_dat["edy"])
        jac_effects_num = jac_effects_num.reshape(jac_effects.shape)
        atol = 10 ** (-4)  # instead of default 10**(-8)
        print("Numeric and algebraic gradient of total effects wrt. direct effects allclose: {}."
              .format(allclose(jac_effects_num, jac_effects, atol=atol)))

    # algebraic delta method effects covariance Matrix
    vcm_effects = jac_effects @ vcm_direct_hat @ jac_effects.T
    effects_std = diag(vcm_effects) ** (1 / 2)
    ex_std, ey_std = directmat_alg(effects_std, model_dat["edx"], model_dat["edy"])
    # set main diag of ey_std to 0, since edy diag is 1 instead of 0
    np.fill_diagonal(ey_std, 0)

    return ex_std, ey_std


def scale(drawing, scaling_factor):
    """Scale a reportlab.graphics.shapes.Drawing() object while maintaining the aspect ratio"""

    drawing.width = drawing.minWidth() * scaling_factor
    drawing.height = drawing.height * scaling_factor
    drawing.scale(scaling_factor, scaling_factor)

    return drawing


def scale_height(drawing, height_cm):
    """scale height in cm,
    limit scaling factor to max DINA4 width minus margins"""

    # convert height_cm to height_dpi: 72 dpi = 1 inch = 2.54 cm
    height_dpi = 72 / 2.54 * height_cm

    scaling_factor = height_dpi / drawing.height

    # limit scaling factor to max_width_dpi
    max_width_dpi = 430
    if drawing.minWidth() * scaling_factor > max_width_dpi:
        scaling_factor = max_width_dpi / drawing.minWidth()

    return scale(drawing, scaling_factor)


def render_dot(dot_str, out_type=None):
    """render Graphviz graph from dot_str to svg or other formats using pydot"""

    if out_type == "svg":
        # avoid svg UTF-8 problems for german umlauts
        dot_str = ''.join([i if ord(i) < 128 else "&#%s;" % ord(i) for i in dot_str])
        graph = pydot.graph_from_dot_data(dot_str)[0]
        xml_string = graph.create_svg()
        graph = svg.fromstring(xml_string)
    else:
        graph = pydot.graph_from_dot_data(dot_str)[0]

    return graph


def save_graph(path, filename, graph_dot):
    """save graph to file as dot string and png"""

    # with open(path + filename + ".txt", "w") as file:
    #    file.write(graph_dot)

    graph = render_dot(graph_dot)
    graph.write_png(path + filename + ".png")

    return


def acc(n1, n2):
    """accuracy: similarity of two numeric matrices,
    between zero (bad) and one (good)"""

    n1 = array(n1)
    n2 = array(n2)
    if norm(n1 - n2) != 0 and norm(n1 + n2) == 0:
        accuracy = 0
    elif norm(n1 - n2) == 0 and norm(n1 + n2) == 0:
        accuracy = 1
    else:
        accuracy = 1 - norm(n1 - n2) / norm(n1 + n2)

    return accuracy
