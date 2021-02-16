# -*- coding: utf-8 -*-
"""Estimation and merging identified linear models over total effects."""

# pylint: disable=invalid-name
# pylint: disable=len-as-condition

import numdifftools as nd
import numpy as np
from copy import deepcopy
from numpy import allclose, array_equal, diag, eye, linspace, zeros
from numpy.linalg import cholesky, inv, LinAlgError

from causing import utils


def sse_hess_num(mx, my, model_dat):
    """compute numeric Hessian of sse at given data and direct effects"""

    def sse_orig_alg(direct, model_dat):
        direct = np.array(direct).reshape(-1)
        mx, my = utils.directmat_alg(direct, model_dat["idx"], model_dat["idy"])
        ex_hat, _ = utils.total_effects_alg(
            mx, my, model_dat["edx"], model_dat["edy"])
        ychat = ex_hat @ model_dat["xcdat"]
        ymchat = model_dat["fym"] @ ychat
        err = ymchat - model_dat["ymcdat"]
        sse = np.sum(err * err * diag(model_dat["selwei"]).reshape(-1, 1))
        ssetikh = sse + model_dat["alpha"] * direct.T @ direct
        return ssetikh

    direct = utils.directvec(mx, my, model_dat["idx"], model_dat["idy"])
    direct = direct.detach().numpy()
    hessian_num_func = nd.Hessian(lambda *direct: sse_orig_alg(direct, model_dat))
    hessian_num = hessian_num_func(direct)

    return hessian_num


def sse_hess_alg(direct_hat, model_dat):
    """compute algebraic Hessian of sse at given data and direct effects

    if called from minimize as hess:
        gets same args passed as sse vec for sse term and Tikhonov term,
    note: sse hess depends on exogeneous data and on direct effects,
          but it does not depend explicitely on tau nor requires a minimum tau
    keyword: target/gradient/hessian function
    """

    mx, my = utils.directmat_alg(direct_hat, model_dat["idx"], model_dat["idy"])

    # define matrices for computation of Hessian
    xcdatxcdatT = model_dat["xcdat"] @ model_dat["xcdat"].T
    ymcdatxcdatT = model_dat["ymcdat"] @ model_dat["xcdat"].T
    i_ = inv(eye(model_dat["ndim"]) - my)
    a_ = model_dat["fym"] @ i_
    e_ = model_dat["fym"].T @ model_dat["selwei"] @ ymcdatxcdatT @ mx.T
    g_ = model_dat["fym"].T @ model_dat["selwei"] @ model_dat["fym"]
    h_ = mx @ xcdatxcdatT @ mx.T

    # define further resused matrix products
    gihiT = g_ @ i_ @ h_ @ i_.T
    iTg = i_.T @ g_
    hiT = h_ @ i_.T
    iTgih = i_.T @ g_ @ i_ @ h_
    iTe = i_.T @ e_
    eiT = e_ @ i_.T

    hessian_sse = zeros((model_dat["qdim"], model_dat["qdim"]))
    # loop quadrant, row wise [[0, 1], [2, 3]]
    for quad in range(4):
        if quad == 0:
            qrowstart = 0
            qcolstart = 0
            irows = model_dat["ndim"]
            jcols = model_dat["ndim"]
            krows = model_dat["ndim"]
            lcols = model_dat["ndim"]
            qrows = model_dat["qydim"]
        if quad == 1:
            qrowstart = 0
            qcolstart = model_dat["qydim"]
            irows = model_dat["ndim"]
            jcols = model_dat["mdim"]
            krows = model_dat["ndim"]
            lcols = model_dat["ndim"]
            qrows = model_dat["qydim"]
        if quad == 2:
            qrowstart = model_dat["qydim"]
            qcolstart = 0
            irows = model_dat["ndim"]
            jcols = model_dat["ndim"]
            krows = model_dat["ndim"]
            lcols = model_dat["mdim"]
            qrows = model_dat["qxdim"]
        if quad == 3:
            qrowstart = model_dat["qydim"]
            qcolstart = model_dat["qydim"]
            irows = model_dat["ndim"]
            jcols = model_dat["mdim"]
            krows = model_dat["ndim"]
            lcols = model_dat["mdim"]
            qrows = model_dat["qxdim"]
        # i, j correspond to cols of Hessian, denominator of derivative
        # iterating column wise, corresponding to vec of direct effects
        qcol = qcolstart
        for j in range(jcols):
            for i in range(irows):
                if quad == 0:
                    jijy = zeros((model_dat["ndim"], model_dat["ndim"]))
                    jijy[i, j] = 1
                    f_ = i_ @ jijy @ i_
                    hess = 2 * ((f_.T @ gihiT + iTg @ f_ @ hiT + iTgih @ f_.T)
                                - (iTe @ f_.T + f_.T @ eiT))
                if quad == 1:
                    jijx = zeros((model_dat["ndim"], model_dat["mdim"]))
                    jijx[i, j] = 1
                    d_ = jijx @ xcdatxcdatT @ mx.T
                    hess = 2 * (a_.T @ model_dat["selwei"] @
                                (a_ @ (d_.T + d_) - ymcdatxcdatT @ jijx.T) @ i_.T)
                if quad == 2:
                    jijy = zeros((model_dat["ndim"], model_dat["ndim"]))
                    jijy[i, j] = 1
                    c_ = model_dat["selwei"] @ a_ @ jijy @ i_
                    hess = 2 * ((a_.T @ c_ + c_.T @ a_) @ mx @ xcdatxcdatT
                                - c_.T @ ymcdatxcdatT)
                if quad == 3:
                    jijx = zeros((model_dat["ndim"], model_dat["mdim"]))
                    jijx[i, j] = 1
                    b_ = model_dat["selwei"] @ a_ @ jijx
                    hess = 2 * (a_.T @ b_ @ xcdatxcdatT)
                # k, l correspond to rows of Hessian, numerator of derivative
                # iterating column wise, corresponding to vec of direct effects
                qrow = qrowstart
                for l in range(lcols):
                    for k in range(krows):
                        if ((quad == 0 and model_dat["idy"][k, l] and model_dat["idy"][i, j])
                                or (quad == 1 and model_dat["idy"][k, l] and model_dat["idx"][i, j])
                                or (quad == 2 and model_dat["idx"][k, l] and model_dat["idy"][i, j])
                                or (quad == 3 and model_dat["idx"][k, l] and model_dat["idx"][i, j])):
                            hessian_sse[qrow, qcol] = hess[k, l]
                            qrow += 1
                if qrow == qrowstart + qrows:
                    qrow = qrowstart
                    qcol += 1

    # Hessian with tikhonov term
    hessian = hessian_sse + 2 * model_dat["alpha"] * eye(model_dat["qdim"])

    # symmetrize Hessian, such that numerically well conditioned
    hessian = (hessian + hessian.T) / 2

    return hessian


def check_hessian(hessian_hat):
    """check algebraic Hessian matrix of target function with respect to
    direct effects at given data and estimated direct effects"""

    # check Hessian symmetric
    if not array_equal(hessian_hat, hessian_hat.T):
        print("-> Hessian not well conditioned: Not symmetric.")
        return False

    # check Hessian positive-definite using Cholesky decomposition
    try:
        cholesky(hessian_hat)
    except LinAlgError:
        # print("-> Hessian not well conditioned: Not positive-definite.")
        return False

    return True


def compute_cov_direct(sse_hat, hessian_hat, model_dat):
    """compute covariance matrix of direct effects"""

    resvar = sse_hat / (model_dat["tau"] - model_dat["dof"])  # yyy
    cov_direct = 2 * resvar * inv(hessian_hat)

    return cov_direct


def check_estimate_effects(model_dat, do_print=True):
    """estimate structural model given alpha in model_dat"""

    mx_hat, my_hat, sse_hat = utils.estimate_snn(model_dat, do_print)

    ex_hat, ey_hat = utils.total_effects_alg(mx_hat, my_hat, model_dat["edx"], model_dat["edy"])
    direct_hat = utils.directvec(mx_hat, my_hat, model_dat["idx"], model_dat["idy"])

    hessian_hat = sse_hess_alg(direct_hat, model_dat)
    check = check_hessian(hessian_hat)
    if check and do_print:
        print("Hessian is well conditioned.")

    return check, hessian_hat, direct_hat, sse_hat, mx_hat, my_hat, ex_hat, ey_hat


def alpha_min_max(model_dat):
    """estimate minimal alpha ensuring positive-definite Hessian
    and give maximal alpha to search over
    
    starting at regularization tikh (= alpha * directnorm)
    being a certain fraction of observed y variance."""

    # alpha_max_tmp
    fraction = 0.002  # ToDo: define globally
    ymvar = np.sum(model_dat["ymcdat"] * model_dat["ymcdat"] *
                   diag(model_dat["selwei"]).reshape(-1, 1))
    directnorm = model_dat["direct_theo"].T @ model_dat["direct_theo"]
    alpha_max_tmp = fraction * ymvar / directnorm

    # try without regularization
    model_dat["alpha"] = 0
    check, *_ = check_estimate_effects(model_dat, do_print=True)
    if check:
        print("\nModel identified without regularization.")
        return 0, alpha_max_tmp
    else:
        print("Hessian not well conditioned at alpha = {}."
              .format(model_dat["alpha"]))

    # regularization
    rel = 0.01  # ToDo: define globally
    absol = min(1e-10, alpha_max_tmp / 1000)  # ToDo: define globally
    alpha_min_tmp = 0
    alpha = (alpha_min_tmp + alpha_max_tmp) / 2
    alpha_min = None
    alpha_max = alpha_max_tmp
    print("\nEstimation of minimal regularization parameter alpha:")
    while (alpha_max_tmp - alpha_min_tmp) / alpha > rel and alpha > absol:
        model_dat["alpha"] = alpha
        check, *_ = check_estimate_effects(model_dat, do_print=False)
        print("alpha: {:10f}, Hessian OK: {}".format(alpha, bool(check)))
        # accept new alpha if Hessian is well conditioned
        if check is False:
            alpha_min_tmp = alpha
            if not alpha_min:  # no alpha found yet
                alpha_max_tmp *= 10
                alpha_max = alpha_max_tmp
        else:
            alpha_max_tmp = alpha
            alpha_min = alpha
        alpha = (alpha_min_tmp + alpha_max_tmp) / 2

    assert alpha_min, "No valid regularization parameter alpha found."

    return alpha_min, alpha_max


def estimate_alpha(alpha_min, alpha_max, model_dat):
    """estimate optimal alpha minimizing out-of-sample SSE via grid search"""

    model_dat_train = deepcopy(model_dat)

    inrel = 0.7  # percentage of in-sample training observations
    num = 10  # number of alphas to search over

    # for in-sample and out-of-sample SSE
    inabs = int(inrel * model_dat["tau"])
    xc_in = model_dat_train["xcdat"][:, :inabs]
    ymc_in = model_dat_train["ymcdat"][:, :inabs]
    xc_out = model_dat_train["xcdat"][:, inabs:]
    ymc_out = model_dat_train["ymcdat"][:, inabs:]

    # for in-sample estimation
    model_dat_train["xcdat"] = model_dat_train["xcdat"][:, :inabs]
    model_dat_train["ymcdat"] = model_dat_train["ymcdat"][:, :inabs]

    found_alpha = False
    while not found_alpha:
        print("\nalpha_min, alpha_max to search over: [{:10f} {:10f}]"
              .format(alpha_min, alpha_max))
        alphas = linspace(alpha_min, alpha_max, num=num)
        mses_ok = []
        alphas_ok = []
        dofs_ok = []
        for alpha in alphas:
            model_dat_train["alpha"] = alpha
            (check, _, _, _, _, _, ex_hat, _
             ) = check_estimate_effects(model_dat_train, do_print=False)  # in-sample train data

            # ToDo: sse in-sample and sse out-of-sample depend on how big companies are:
            # use base_var for data normalization before estimation # yyyy

            # in-sample mse
            ychat_in = ex_hat @ xc_in
            ymchat_in = model_dat_train["fym"] @ ychat_in
            err_in = ymchat_in - ymc_in
            sse_in = np.sum(err_in * err_in * diag(model_dat_train["selwei"]).reshape(-1, 1))
            mse_in = sse_in / inabs

            # in-sample mse, central
            err_central_in = err_in - np.mean(err_in, axis=1).reshape(-1, 1)
            sse_central_in = np.sum(err_central_in * err_central_in * diag(model_dat_train["selwei"]).reshape(-1, 1))
            mse_central_in = sse_central_in / inabs

            # out-of-sample mse, for out-of-sample test data
            ychat = ex_hat @ xc_out
            ymchat = model_dat_train["fym"] @ ychat
            err = ymchat - ymc_out
            sse = np.sum(err * err * diag(model_dat_train["selwei"]).reshape(-1, 1))
            mse = sse / (model_dat["tau"] - inabs)

            # dof, Tibshirani (2015), "Degrees of Freedom and Model Search", eq. (5)
            dof = (mse - mse_in) / (2 * mse_central_in)
            dof = min(max(dof, 0), model_dat["qdim"])

            if check:
                mses_ok.append(sse)
                alphas_ok.append(alpha)
                dofs_ok.append(dof)
            print("alpha: {:10f}, Hessian OK: {:5s}, out-of-sample mse: {:10f}, dof: {:10f}"
                  .format(alpha, str(bool(check)), mse, dof))

        # check that full data Hessian is also positive-definite
        # sort by mses_ok
        if len(alphas_ok) > 0:
            mses_ok, alphas_ok, dofs_ok = zip(*sorted(zip(mses_ok, alphas_ok, dofs_ok)))
            print("\ncheck alpha with full data:")
            for i, alpha in enumerate(alphas_ok):
                model_dat["alpha"] = alpha
                check, *_ = check_estimate_effects(model_dat, do_print=False)  # full data
                dof = dofs_ok[i]
                print("alpha: {:10f}, dof: {:10f}, Hessian OK: {:5s}"
                      .format(alpha, dof, str(bool(check))))
                if check:
                    break

        # no alpha found or optimal alpha is alpha_max
        if not check or alpha == alpha_max:
            print("Increase alpha_max.")
            alpha_max *= 10
            found_alpha = False
        else:
            found_alpha = True

    print("optimal alpha with minimal out-of-sample sse: {:10f}, dof: {:10f}"
          .format(alpha, dof))

    return alpha, dof


def estimate_effects(model_dat):
    """nonlinear estimation of linearized structural model
    using theoretical direct effects as starting values"""  # ToDo: reintroduce # yyyy

    if model_dat["alpha"] is None:
        if model_dat["dof"] is not None:
            raise ValueError("dof is determined together with alpha.")

        # alpha_min (with posdef hessian) and alpha_max to search over
        alpha_min, alpha_max = alpha_min_max(model_dat)

        # optimal alpha with minimal out-of-sample sse
        alpha, dof = estimate_alpha(alpha_min, alpha_max, model_dat)
        model_dat["alpha"] = alpha
        model_dat["dof"] = dof
    else:
        if model_dat["dof"] is None:
            raise ValueError("dof must be given together with alpha.")
        print("\ngiven alpha: {:10f}, dof: {:10f}"
              .format(model_dat["alpha"], model_dat["dof"]))

    # final estimation given optimal alpha
    # algebraic Hessian
    (check, hessian_hat, direct_hat, sse_hat, mx_hat, my_hat, ex_hat, ey_hat
     ) = check_estimate_effects(model_dat)
    # automatic Hessian
    hessian = utils.sse_hess(mx_hat, my_hat, model_dat)
    # numeric Hessian
    hessian_num = sse_hess_num(mx_hat, my_hat, model_dat)

    print("\nAlgebraic and numeric   Hessian allclose: {} with accuracy {:10f}."
          .format(allclose(hessian_hat, hessian_num),
                  utils.acc(hessian_hat, hessian_num)))
    print("Automatic and numeric   Hessian allclose: {} with accuracy {:10f}."
          .format(allclose(hessian, hessian_num),
                  utils.acc(hessian, hessian_num)))
    print("Automatic and algebraic Hessian allclose: {} with accuracy {:10f}."
          .format(allclose(hessian, hessian_hat),
                  utils.acc(hessian, hessian_hat)))

    assert check, "Hessian not well conditioned."
    cov_direct_hat = compute_cov_direct(sse_hat, hessian_hat, model_dat)

    # compute estimated direct, total and mediation effects and standard deviations
    mx_hat_std, my_hat_std = utils.compute_direct_std(cov_direct_hat, model_dat)
    ex_hat_std, ey_hat_std = utils.total_effects_std(direct_hat, cov_direct_hat, model_dat)
    exj_hat, eyj_hat, eyx_hat, eyy_hat = utils.compute_mediation_effects(
        mx_hat, my_hat, ex_hat, ey_hat, model_dat["yvars"], model_dat["final_var"])
    (exj_hat_std, eyj_hat_std, eyx_hat_std, eyy_hat_std
     ) = utils.compute_mediation_std(ex_hat_std, ey_hat_std, eyx_hat, eyy_hat,
                                     model_dat["yvars"], model_dat["final_var"])

    estimate_dat = {
        "direct_hat": direct_hat,
        "sse_hat": sse_hat,
        "hessian_hat": hessian_hat,
        "cov_direct_hat": cov_direct_hat,
        "mx_hat": mx_hat,
        "my_hat": my_hat,
        "mx_hat_std": mx_hat_std,
        "my_hat_std": my_hat_std,
        "ex_hat": ex_hat,
        "ey_hat": ey_hat,
        "ex_hat_std": ex_hat_std,
        "ey_hat_std": ey_hat_std,
        "exj_hat": exj_hat,
        "eyj_hat": eyj_hat,
        "eyx_hat": eyx_hat,
        "eyy_hat": eyy_hat,
        "exj_hat_std": exj_hat_std,
        "eyj_hat_std": eyj_hat_std,
        "eyx_hat_std": eyx_hat_std,
        "eyy_hat_std": eyy_hat_std,
    }

    return estimate_dat


def estimate_biases(model_dat):
    """numerical optimize modification indicators for equations, one at a time"""

    biases = zeros(model_dat["ndim"])
    biases_std = zeros(model_dat["ndim"])
    for bias_ind in range(model_dat["ndim"]):
        # compute biases
        bias, hess_i, sse = utils.optimize_biases(model_dat, bias_ind)
        biases[bias_ind] = bias

        # compute biases_std
        resvar = sse / (model_dat["tau"] - 1)
        bias_std = (2 * resvar * (1 / hess_i)) ** (1 / 2)
        biases_std[bias_ind] = bias_std

    return biases, biases_std


def estimate_models(model_dat):
    """estimation of modification indicators of level model"""

    # estimate linear models
    estimate_dat = estimate_effects(model_dat)

    # estimate equation biases, given theoretical level model
    if model_dat["estimate_bias"]:
        biases, biases_std = estimate_biases(model_dat)
        estimate_dat["biases"] = biases
        estimate_dat["biases_std"] = biases_std

    return estimate_dat
