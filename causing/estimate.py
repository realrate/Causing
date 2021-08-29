# -*- coding: utf-8 -*-
"""Estimation and merging identified linear models over total effects."""

# pylint: disable=invalid-name
# pylint: disable=len-as-condition

from copy import copy, deepcopy

import numdifftools as nd
import numpy as np
from numpy import allclose, array_equal, diag, eye, linspace, zeros, isnan
from numpy.linalg import cholesky, inv, LinAlgError
import torch
from scipy.optimize import minimize

from causing import utils


def sse_hess_num(mx, my, model_dat, alpha):
    """compute numeric Hessian of sse at given data and direct effects"""

    def sse_orig_alg(direct, model_dat):
        direct = np.array(direct).reshape(-1)
        mx, my = utils.directmat_alg(direct, model_dat["idx"], model_dat["idy"])
        ex_hat, _ = utils.total_effects_alg(mx, my, model_dat["edx"], model_dat["edy"])
        ex_hat = utils.nan_to_zero(ex_hat)
        ychat = ex_hat @ model_dat["xcdat"]
        ymchat = model_dat["fym"] @ ychat
        err = ymchat - model_dat["ymcdat"]
        sse = np.sum(err * err * diag(model_dat["selwei"]).reshape(-1, 1))
        ssetikh = sse + alpha * direct.T @ direct
        return ssetikh

    direct = utils.directvec(mx, my, model_dat["idx"], model_dat["idy"])
    direct = direct.detach().numpy()
    hessian_num_func = nd.Hessian(lambda *direct: sse_orig_alg(direct, model_dat))
    hessian_num = hessian_num_func(direct)

    return hessian_num


def sse_hess_alg(direct_hat, model_dat, alpha):
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
                    hess = 2 * (
                        (f_.T @ gihiT + iTg @ f_ @ hiT + iTgih @ f_.T)
                        - (iTe @ f_.T + f_.T @ eiT)
                    )
                if quad == 1:
                    jijx = zeros((model_dat["ndim"], model_dat["mdim"]))
                    jijx[i, j] = 1
                    d_ = jijx @ xcdatxcdatT @ mx.T
                    hess = 2 * (
                        a_.T
                        @ model_dat["selwei"]
                        @ (a_ @ (d_.T + d_) - ymcdatxcdatT @ jijx.T)
                        @ i_.T
                    )
                if quad == 2:
                    jijy = zeros((model_dat["ndim"], model_dat["ndim"]))
                    jijy[i, j] = 1
                    c_ = model_dat["selwei"] @ a_ @ jijy @ i_
                    hess = 2 * (
                        (a_.T @ c_ + c_.T @ a_) @ mx @ xcdatxcdatT - c_.T @ ymcdatxcdatT
                    )
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
                        if (
                            (
                                quad == 0
                                and model_dat["idy"][k, l]
                                and model_dat["idy"][i, j]
                            )
                            or (
                                quad == 1
                                and model_dat["idy"][k, l]
                                and model_dat["idx"][i, j]
                            )
                            or (
                                quad == 2
                                and model_dat["idx"][k, l]
                                and model_dat["idy"][i, j]
                            )
                            or (
                                quad == 3
                                and model_dat["idx"][k, l]
                                and model_dat["idx"][i, j]
                            )
                        ):
                            hessian_sse[qrow, qcol] = hess[k, l]
                            qrow += 1
                if qrow == qrowstart + qrows:
                    qrow = qrowstart
                    qcol += 1

    # Hessian with tikhonov term
    hessian = hessian_sse + 2 * alpha * eye(model_dat["qdim"])

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


def compute_cov_direct(sse_hat, hessian_hat, model_dat, dof):
    """compute covariance matrix of direct effects"""

    tau = model_dat["xdat"].shape[1]
    resvar = sse_hat / (tau - dof)  # yyy
    cov_direct = 2 * resvar * inv(hessian_hat)

    return cov_direct


def check_estimate_effects(model_dat, alpha, do_print=True):
    """estimate structural model given alpha in model_dat"""

    mx_hat, my_hat, sse_hat = estimate_snn(model_dat, alpha, do_print)
    mx_hat[model_dat["idx"] == 0] = float("NaN")
    my_hat[model_dat["idy"] == 0] = float("NaN")

    ex_hat, ey_hat = utils.total_effects_alg(
        mx_hat, my_hat, model_dat["edx"], model_dat["edy"]
    )
    ex_hat[model_dat["edx"] == 0] = float("NaN")
    ey_hat[model_dat["edy"] == 0] = float("NaN")
    direct_hat = utils.directvec(mx_hat, my_hat, model_dat["idx"], model_dat["idy"])

    hessian_hat = sse_hess_alg(direct_hat, model_dat, alpha)
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
    ymvar = np.sum(
        model_dat["ymcdat"]
        * model_dat["ymcdat"]
        * diag(model_dat["selwei"]).reshape(-1, 1)
    )
    direct_theo = utils.directvec_alg(model_dat["mx_theo"], model_dat["my_theo"])
    directnorm = direct_theo.T @ direct_theo
    alpha_max_tmp = fraction * ymvar / directnorm

    # try without regularization
    alpha = 0
    check, *_ = check_estimate_effects(model_dat, alpha, do_print=True)
    if check:
        print("\nModel identified without regularization.")
        return 0, alpha_max_tmp
    else:
        print("Hessian not well conditioned at alpha = {}.".format(alpha))

    # regularization
    rel = 0.01  # ToDo: define globally
    absol = min(1e-10, alpha_max_tmp / 1000)  # ToDo: define globally
    alpha_min_tmp = 0
    alpha = (alpha_min_tmp + alpha_max_tmp) / 2
    alpha_min = None
    alpha_max = alpha_max_tmp
    print("\nEstimation of minimal regularization parameter alpha:")
    while (alpha_max_tmp - alpha_min_tmp) / alpha > rel and alpha > absol:
        check, *_ = check_estimate_effects(model_dat, alpha, do_print=False)
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

    tau = model_dat["xdat"].shape[1]
    model_dat_train = deepcopy(model_dat)

    inrel = 0.7  # percentage of in-sample training observations
    num = 10  # number of alphas to search over

    # for in-sample and out-of-sample SSE
    inabs = int(inrel * tau)
    xc_in = model_dat_train["xcdat"][:, :inabs]
    ymc_in = model_dat_train["ymcdat"][:, :inabs]
    xc_out = model_dat_train["xcdat"][:, inabs:]
    ymc_out = model_dat_train["ymcdat"][:, inabs:]

    # for in-sample estimation
    model_dat_train["xcdat"] = model_dat_train["xcdat"][:, :inabs]
    model_dat_train["ymcdat"] = model_dat_train["ymcdat"][:, :inabs]

    found_alpha = False
    while not found_alpha:
        print(
            "\nalpha_min, alpha_max to search over: [{:10f} {:10f}]".format(
                alpha_min, alpha_max
            )
        )
        alphas = linspace(alpha_min, alpha_max, num=num)
        mses_ok = []
        alphas_ok = []
        dofs_ok = []
        for alpha in alphas:
            (check, _, _, _, _, _, ex_hat, _) = check_estimate_effects(
                model_dat_train, alpha, do_print=False
            )  # in-sample train data
            ex_hat = utils.nan_to_zero(ex_hat)

            # ToDo: sse in-sample and sse out-of-sample depend on how big companies are:
            # use base_var for data normalization before estimation # yyyy

            # in-sample mse
            ychat_in = ex_hat @ xc_in
            ymchat_in = model_dat_train["fym"] @ ychat_in
            err_in = ymchat_in - ymc_in
            sse_in = np.sum(
                err_in * err_in * diag(model_dat_train["selwei"]).reshape(-1, 1)
            )
            mse_in = sse_in / inabs

            # in-sample mse, central
            err_central_in = err_in - np.mean(err_in, axis=1).reshape(-1, 1)
            sse_central_in = np.sum(
                err_central_in
                * err_central_in
                * diag(model_dat_train["selwei"]).reshape(-1, 1)
            )
            mse_central_in = sse_central_in / inabs

            # out-of-sample mse, for out-of-sample test data
            ychat = ex_hat @ xc_out
            ymchat = model_dat_train["fym"] @ ychat
            err = ymchat - ymc_out
            sse = np.sum(err * err * diag(model_dat_train["selwei"]).reshape(-1, 1))
            mse = sse / (tau - inabs)

            # dof, Tibshirani (2015), "Degrees of Freedom and Model Search", eq. (5)
            dof = (mse - mse_in) / (2 * mse_central_in)
            dof = min(max(dof, 0), model_dat["qdim"])

            if check:
                mses_ok.append(sse)
                alphas_ok.append(alpha)
                dofs_ok.append(dof)
            print(
                "alpha: {:10f}, Hessian OK: {:5s}, out-of-sample mse: {:10f}, dof: {:10f}".format(
                    alpha, str(bool(check)), mse, dof
                )
            )

        # check that full data Hessian is also positive-definite
        # sort by mses_ok
        if len(alphas_ok) > 0:
            mses_ok, alphas_ok, dofs_ok = zip(*sorted(zip(mses_ok, alphas_ok, dofs_ok)))
            print("\ncheck alpha with full data:")
            for i, alpha in enumerate(alphas_ok):
                check, *_ = check_estimate_effects(
                    model_dat, alpha, do_print=False
                )  # full data
                dof = dofs_ok[i]
                print(
                    "alpha: {:10f}, dof: {:10f}, Hessian OK: {:5s}".format(
                        alpha, dof, str(bool(check))
                    )
                )
                if check:
                    break

        # no alpha found or optimal alpha is alpha_max
        if not check or alpha == alpha_max:
            print("Increase alpha_max.")
            alpha_max *= 10
            found_alpha = False
        else:
            found_alpha = True

    print(
        "optimal alpha with minimal out-of-sample sse: {:10f}, dof: {:10f}".format(
            alpha, dof
        )
    )

    return alpha, dof


def estimate_effects(model_dat, estimate_input):
    """nonlinear estimation of linearized structural model
    using theoretical direct effects as starting values"""  # ToDo: reintroduce # yyyy

    # PyTorch does not like NaNs, so create a copy of model_dat with zeros
    # instead of NaNs for start values
    model_dat = deepcopy(model_dat)
    for key in ["mx_theo", "my_theo"]:
        model_dat[key] = utils.nan_to_zero(model_dat[key])

    if estimate_input["alpha"] is None:
        if estimate_input["dof"] is not None:
            raise ValueError("dof is determined together with alpha.")

        # alpha_min (with posdef hessian) and alpha_max to search over
        alpha_min, alpha_max = alpha_min_max(model_dat)

        # optimal alpha with minimal out-of-sample sse
        alpha, dof = estimate_alpha(alpha_min, alpha_max, model_dat)
        estimate_input["alpha"] = alpha
        estimate_input["dof"] = dof
    else:
        if estimate_input["dof"] is None:
            raise ValueError("dof must be given together with alpha.")
        print(
            "\ngiven alpha: {:10f}, dof: {:10f}".format(
                estimate_input["alpha"], estimate_input["dof"]
            )
        )

    # final estimation given optimal alpha
    # algebraic Hessian
    (
        check,
        hessian_hat,
        direct_hat,
        sse_hat,
        mx_hat,
        my_hat,
        ex_hat,
        ey_hat,
    ) = check_estimate_effects(model_dat, estimate_input["alpha"])
    # automatic Hessian
    hessian = sse_hess(mx_hat, my_hat, model_dat, estimate_input["alpha"])
    # numeric Hessian
    hessian_num = sse_hess_num(mx_hat, my_hat, model_dat, estimate_input["alpha"])

    print(
        "\nAlgebraic and numeric   Hessian allclose: {} with accuracy {:10f}.".format(
            allclose(hessian_hat, hessian_num), utils.acc(hessian_hat, hessian_num)
        )
    )
    print(
        "Automatic and numeric   Hessian allclose: {} with accuracy {:10f}.".format(
            allclose(hessian, hessian_num), utils.acc(hessian, hessian_num)
        )
    )
    print(
        "Automatic and algebraic Hessian allclose: {} with accuracy {:10f}.".format(
            allclose(hessian, hessian_hat), utils.acc(hessian, hessian_hat)
        )
    )

    assert check, "Hessian not well conditioned."
    cov_direct_hat = compute_cov_direct(
        sse_hat, hessian_hat, model_dat, estimate_input["dof"]
    )

    # compute estimated direct, total and mediation effects and standard deviations
    mx_hat_std, my_hat_std = compute_direct_std(cov_direct_hat, model_dat)
    ex_hat_std, ey_hat_std = total_effects_std(direct_hat, cov_direct_hat, model_dat)
    exj_hat, eyj_hat, eyx_hat, eyy_hat = utils.compute_mediation_effects(
        mx_hat, my_hat, ex_hat, ey_hat, model_dat["yvars"], model_dat["final_var"]
    )
    (exj_hat_std, eyj_hat_std, eyx_hat_std, eyy_hat_std) = compute_mediation_std(
        ex_hat_std,
        ey_hat_std,
        eyx_hat,
        eyy_hat,
        model_dat["yvars"],
        model_dat["final_var"],
    )
    # final_ind = list(model_dat["yvars"]).index(model_dat["final_var"])
    # eyx_hat[model_dat["edx"] == 0] = float("NaN")
    # eyy_hat[model_dat["edy"][final_ind] == 0] = float("NaN")

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


def estimate_biases(model_dat, ymdat):
    """numerical optimize modification indicators for equations, one at a time"""

    tau = model_dat["xdat"].shape[1]
    biases = zeros(model_dat["ndim"])
    biases_std = zeros(model_dat["ndim"])
    for bias_ind in range(model_dat["ndim"]):
        # compute biases
        bias, hess_i, sse = optimize_biases(model_dat, bias_ind, ymdat)
        biases[bias_ind] = bias

        # compute biases_std
        resvar = sse / (tau - 1)
        bias_std = (2 * resvar * (1 / hess_i)) ** (1 / 2)
        biases_std[bias_ind] = bias_std

    return biases, biases_std


def estimate_models(m, xdat, mean_theo, estimate_input):
    """estimation of modification indicators of level model"""

    model_dat = {  # TODO: completely remove model_dat
        # from Model class
        "m": m,
        "ndim": m.ndim,
        "mdim": m.mdim,
        "pdim": len(m.ymvars),
        "qxdim": m.qxdim,
        "qydim": m.qydim,
        "qdim": m.qdim,
        "idx": m.idx,
        "idy": m.idy,
        "edx": m.edx,
        "edy": m.edy,
        "fdx": m.fdx,
        "fdy": m.fdy,
        "model": m.compute,
        "mx_lam": m.m_pair[0],
        "my_lam": m.m_pair[1],
        "xvars": m.xvars,
        "yvars": m.yvars,
        "final_var": m.final_var,
        "ymvars": m.ymvars,
        # other
        "xdat": xdat,
        "tau": xdat.shape[1],
    }
    model_dat.update(mean_theo)

    # check
    if estimate_input["ymdat"].shape[0] != model_dat["pdim"]:
        raise ValueError(
            "Number of ymvars {} and ymdat {} not identical.".format(
                estimate_input["ymdat"].shape[0], model_dat["pdim"]
            )
        )

    xmean = model_dat["xdat"].mean(axis=1)
    xcdat = model_dat["xdat"] - xmean.reshape(model_dat["mdim"], 1)
    ymmean = estimate_input["ymdat"].mean(axis=1)
    ymcdat = estimate_input["ymdat"] - ymmean.reshape(model_dat["pdim"], 1)

    selvec = zeros(model_dat["ndim"])
    selvec[[list(model_dat["yvars"]).index(el) for el in model_dat["ymvars"]]] = 1
    selmat = diag(selvec)
    # selwei whitening matrix of manifest demeaned variables
    selwei = diag(1 / np.var(ymcdat, axis=1))
    fm = eye(model_dat["ndim"] + model_dat["mdim"])[
        np.concatenate((selvec, np.ones(model_dat["mdim"]))) == 1
    ]
    fym = eye(model_dat["ndim"])[selvec == 1]

    model_dat.update(
        dict(
            fm=fm,
            fym=fym,
            selmat=selmat,
            selwei=selwei,
            xcdat=xcdat,
            ymcdat=ymcdat,
        )
    )

    # estimate linear models
    # estimate_input is updated with estimated alpha, dof
    estimate_dat = estimate_effects(model_dat, estimate_input)

    # estimate equation biases, given theoretical level model
    if estimate_input["estimate_bias"]:
        biases, biases_std = estimate_biases(model_dat, estimate_input["ymdat"])
        estimate_dat["biases"] = biases
        estimate_dat["biases_std"] = biases_std

    # set missing edges to NaN
    estimate_dat["mx_hat"][model_dat["idx"] == 0] = float("NaN")
    estimate_dat["my_hat"][model_dat["idy"] == 0] = float("NaN")
    estimate_dat["mx_hat_std"][model_dat["idx"] == 0] = float("NaN")
    estimate_dat["my_hat_std"][model_dat["idy"] == 0] = float("NaN")

    return estimate_dat


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


def optimize_ssn(
    ad_model,
    mx,
    my,
    fym,
    ydata,
    selwei,
    model_dat,
    optimizer,
    params,
    alpha,
    do_print=True,
):
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
        sse = sse_orig(mx, my, fym, ychat, ydata, selwei, model_dat, alpha)  # forward
        optimizer.zero_grad()
        sse.backward(create_graph=True)  # backward
        optimizer.step()
        if abs(sse - sse_old) / sse_old < rel:
            nr_conv += 1
        else:
            nr_conv = 0
        nrm = sum([torch.norm(param) for param in params]).detach().numpy()  # type: ignore
        if do_print:
            print(
                "epoch {:>4}, sse {:10f}, param norm {:10f}".format(
                    epoch, sse.item(), nrm
                )
            )
        epoch += 1

    return sse


def estimate_snn(model_dat, alpha, do_print=True):
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
        print("\nEstimation of direct effects using a structural neural network.")
    sse = optimize_ssn(
        ad_model,
        mx,
        my,
        fym,
        ydata,
        selwei,
        model_dat,
        optimizer,
        params,
        alpha,
        do_print,
    )

    mx = mx.detach().numpy()
    my = my.detach().numpy()
    sse = sse.detach().numpy()
    assert allclose(
        mx, mx * model_dat["idx"]
    ), "idx identification restrictions not met:\n{}\n!=\n{}".format(
        mx, mx * model_dat["idx"]
    )
    assert allclose(
        my, my * model_dat["idy"]
    ), "idy identification restrictions not met:\n{}\n!=\n{}".format(
        my, my * model_dat["idy"]
    )

    return mx, my, sse


def sse_bias(bias, bias_ind, model_dat, ymdat):
    """sum of squared errors given modification indicator, Tikhonov not used"""
    bias = bias[0]

    yhat = model_dat["model"](model_dat["xdat"], bias, bias_ind)
    ymhat = model_dat["fym"] @ yhat
    err = ymhat - ymdat
    sse = np.sum(err * err * diag(model_dat["selwei"]).reshape(-1, 1))

    print("sse {:10f}, bias {:10f}".format(sse, bias))
    return sse


def optimize_biases(model_dat, bias_ind, ymdat):
    """numerical optimize modification indicator for single equation"""

    # optimizations parameters
    bias_start = 0
    method = "SLSQP"  # BFGS, SLSQP, Nelder-Mead, Powell, TNC, COBYLA, CG

    print("\nEstimation of bias for {}:".format(model_dat["yvars"][bias_ind]))
    out = minimize(
        sse_bias, bias_start, args=(bias_ind, model_dat, ymdat), method=method
    )

    bias = out.x
    sse = out.fun

    if hasattr(out, "hess_inv"):
        hess_i = inv(out.hess_inv)
        print("Scalar Hessian from method {}.".format(method))
    else:
        hess_i = nd.Derivative(sse_bias, n=2)(bias, bias_ind, model_dat, ymdat)
        print("Scalar Hessian numerically.")

    return bias, hess_i, sse


def sse_hess(mx, my, model_dat, alpha):
    """compute automatic Hessian of sse at given data and direct effects"""

    fym = torch.DoubleTensor(model_dat["fym"])
    ydata = torch.DoubleTensor(model_dat["ymcdat"])
    selwei = torch.DoubleTensor(model_dat["selwei"])

    def sse_orig_vec_alg(direct):
        """computes the ad target function sum of squared errors,
        input as tensor vectors, yields Hessian in usual dimension of identified parameters"""
        mx, my = utils.directmat(direct, model_dat["idx"], model_dat["idy"])
        ad_model = StructuralNN(model_dat)
        ychat = ad_model(mx, my)
        return sse_orig(mx, my, fym, ychat, ydata, selwei, model_dat, alpha)

    direct = utils.directvec(mx, my, model_dat["idx"], model_dat["idy"])
    hessian = torch.autograd.functional.hessian(sse_orig_vec_alg, direct)

    # symmetrize Hessian, such that numerically well conditioned
    hessian = hessian.detach().numpy()
    hessian = (hessian + hessian.T) / 2

    return hessian


def tvals(eff, std):
    """compute t-values by element wise division of eff and std matrices"""

    assert eff.shape == std.shape

    if len(eff.shape) == 1:  # vector
        rows = eff.shape[0]
        tvalues = np.empty(rows) * np.nan
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
                    tvalues[i, j] = np.nan

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
    exj_hat_std_mat = np.tile(exj_hat_std, (ndim, 1))  # (ndim x mdim)
    eyj_hat_std_mat = np.tile(eyj_hat_std, (ndim, 1))  # (ndim x ndim)

    # column sums of mediation matrices
    x_colsum = np.nansum(eyx, axis=0)
    y_colsum = np.nansum(eyy, axis=0)
    # normed mediation matrices by division by column sums,
    # zero sum for varaibles w/o effect on others
    # substituted by nan to avoid false interpretation
    x_colsum[x_colsum == 0] = np.nan
    y_colsum[y_colsum == 0] = np.nan
    eyx_colnorm = zeros((ndim, mdim))  # (ndim x mdim)
    eyx_colnorm[:] = np.nan
    for j in range(mdim):
        if not isnan(x_colsum[j]):
            eyx_colnorm[:, j] = eyx[:, j] / x_colsum[j]
    eyy_colnorm = zeros((ndim, ndim))  # (ndim x ndim)
    eyy_colnorm[:] = np.nan
    for j in range(ndim):
        if not isnan(y_colsum[j]):
            eyy_colnorm[:, j] = eyy[:, j] / y_colsum[j]

    # mediation std matrices
    eyx_hat_std = exj_hat_std_mat * eyx_colnorm  # (ndim x mdim)
    eyy_hat_std = eyj_hat_std_mat * eyy_colnorm  # (ndim x ndim)

    return exj_hat_std, eyj_hat_std, eyx_hat_std, eyy_hat_std


def compute_direct_std(vcm_direct_hat, model_dat):
    """compute direct effects standard deviations from direct effects covariance matrix"""

    direct_std = diag(vcm_direct_hat) ** (1 / 2)
    mx_std, my_std = utils.directmat_alg(direct_std, model_dat["idx"], model_dat["idy"])

    return mx_std, my_std


def total_effects_std(direct_hat, vcm_direct_hat, model_dat):
    """compute total effects standard deviations

    given estimated vcm_direct_hat,
    using algebraic delta method for covariance Matrix of effects and
    algebraic gradient of total effects wrt. direct effects
    """

    # compute vec matrices for algebraic effects gradient wrt. to direct_hat
    vecmaty = utils.vecmat(model_dat["idy"])
    vecmatx = utils.vecmat(model_dat["idx"])
    vecmaty = np.hstack(
        (vecmaty, zeros((model_dat["ndim"] * model_dat["ndim"], model_dat["qxdim"])))
    )
    vecmatx = np.hstack(
        (zeros((model_dat["ndim"] * model_dat["mdim"], model_dat["qydim"])), vecmatx)
    )

    # compute algebraic gradient of total effects wrt. direct effects
    mx, my = utils.directmat_alg(direct_hat, model_dat["idx"], model_dat["idy"])
    ey = inv(eye(model_dat["ndim"]) - my)
    jac_effects_y = (
        np.kron(ey.T, ey) - eye(model_dat["ndim"] * model_dat["ndim"])
    ) @ vecmaty + vecmaty
    jac_effects_x = (
        np.kron((ey @ mx).T, ey) @ vecmaty
        + np.kron(eye(model_dat["mdim"]), (ey - eye(model_dat["ndim"]))) @ vecmatx
        + vecmatx
    )

    # reduce to rows corresponding to nonzero effects
    indy = (
        np.reshape(model_dat["edy"], model_dat["ndim"] * model_dat["ndim"], order="F")
        != 0
    )
    indx = (
        np.reshape(model_dat["edx"], model_dat["ndim"] * model_dat["mdim"], order="F")
        != 0
    )
    jac_effects_y = jac_effects_y[indy, :]
    jac_effects_x = jac_effects_x[indx, :]
    jac_effects = np.vstack((jac_effects_y, jac_effects_x))

    # compare numeric and algebraic gradient of effects
    do_compare = True
    if do_compare:
        jac_effects_num = nd.Jacobian(utils.total_from_direct)(
            direct_hat,
            model_dat["idx"],
            model_dat["idy"],
            model_dat["edx"],
            model_dat["edy"],
        )
        jac_effects_num = jac_effects_num.reshape(jac_effects.shape)
        atol = 10 ** (-4)  # instead of default 10**(-8)
        print(
            "Numeric and algebraic gradient of total effects wrt. direct effects allclose: {}.".format(
                allclose(jac_effects_num, jac_effects, atol=atol)
            )
        )

    # algebraic delta method effects covariance Matrix
    vcm_effects = jac_effects @ vcm_direct_hat @ jac_effects.T
    effects_std = diag(vcm_effects) ** (1 / 2)
    ex_std, ey_std = utils.directmat_alg(
        effects_std, model_dat["edx"], model_dat["edy"]
    )
    # set main diag of ey_std to 0, since edy diag is 1 instead of 0
    np.fill_diagonal(ey_std, 0)

    ex_std[model_dat["edx"] == 0] = float("NaN")
    ey_std[model_dat["edy"] == 0] = float("NaN")
    return ex_std, ey_std


def sse_orig(mx, my, fym, ychat, ymcdat, selwei, model_dat, alpha):
    """weighted MSE target function plus Tikhonov regularization term"""

    # weighted mean squared error
    ymchat = fym @ ychat
    err = ymchat - ymcdat

    # sse = torch.trace(err.T @ selwei @ err) # big matrix needs too much RAM
    # elementwise multiplication and broadcasting and summation:
    sse = sum(torch.sum(err * err * torch.diag(selwei).view(-1, 1), dim=0))

    # sse with tikhonov term
    direct = utils.directvec(mx, my, model_dat["idx"], model_dat["idy"])
    ssetikh = sse + alpha * direct.T @ direct

    return ssetikh.requires_grad_(True)


def filter_important_keys(estimate_dat):
    saved_estimate_keys = [
        # EDE json
        "mx_hat",
        "my_hat",
        # EME json
        "eyx_hat",
        "eyy_hat",
        "exj_hat",
        "eyj_hat",
        # ED0_json
        "mx_hat_std",
        "my_hat_std",
        # EM0_json
        "eyx_hat_std",
        "exj_hat_std",
        "eyy_hat_std",
        "eyj_hat_std",
        # ETE json
        "ex_hat",
        "ey_hat",
        # ET0 json
        "ex_hat_std",
        "ey_hat_std",
    ]
    return {k: v for k, v in estimate_dat.items() if k in saved_estimate_keys}
