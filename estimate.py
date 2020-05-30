# -*- coding: utf-8 -*-
"""Estimation and merging identified linear models over total effects."""

# pylint: disable=invalid-name
# pylint: disable=len-as-condition

import numpy as np
from numpy import allclose, array_equal, diag, eye, finfo, trace, zeros
from numpy.linalg import inv, LinAlgError, matrix_rank, svd

import utils


def sse_hess_alg(coeffs_hat, model_dat):
    """compute algebraic Hessian of sse at given data

    if called from minimize as hess:
        gets same args passed as sse vec for sse term and Tikhonov term,
    note: sse hess depends on exogeneous data and on coeffs,
          but it does not depend explicitely on tau nor requires a minimum tau
    keyword: target/gradient/hessian function
    """

    mx, my = utils.coeffmat_alg(coeffs_hat, model_dat["idx"], model_dat["idy"])

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
        # iterating column wise, corresponding to vec of coeffs
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
                # iterating column wise, corresponding to vec of coeffs
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

def check_hessian(sse_hat, hessian, model_dat, do_print=True):
    """check algebraic Hessian matrix of target function at given data"""

    # check symmetric Hessian
    if not array_equal(hessian, hessian.T):
        print("-> Hessian not well conditioned: Not symmetric.")
        return False, None

    # check if Hessian is posdef by Cholesky and eigenvalues
    if not utils.isposdef(hessian):
        print("-> Hessian not well conditioned: Not positive-definite.")
        return False, None

    # check full rank of algebraic Hessian at given data
    rank_hessian = matrix_rank(hessian)
    redrank = model_dat["qdim"] - rank_hessian
    if redrank > 0:
        print("-> Hessian not well conditioned: Reduced rank: {}. ".format(rank_hessian))
        return False, None

    # check invertibility of algebraic Hessian
    try:
        inv_hessian = inv(hessian)
    except LinAlgError:
        print("-> Hessian not well conditioned: LinAlgError: Singular matrix.")
        return False, None

    # check nonnegative coefficient variances given by algebraic Hessian
    varvec = diag(inv_hessian)
    if not np.all(varvec >= 0):
        print("-> Hessian not well conditioned: Inverse yields negative coeff. variances, "
              "min value: {}.".format(min(varvec)))
        return False, None

    # check for small singular values of hessian relative to machine tol
    svd_hessian = svd(hessian, compute_uv=False)
    # tolerance for singular values of svd_hessian accepted as numerically different from zero
    tol = abs(svd_hessian).max() * max(svd_hessian.shape) * finfo(float).eps
    rel_svd_hessian = svd_hessian / tol
    tol_singval = 1 # ToDo: calibrate 10**4
    if any(rel_svd_hessian < tol_singval):
        print("->  not well conditioned: Some singular values are small (< {}) "
              "relative to machine tolerance, min value: {}."
              .format(tol_singval, min(rel_svd_hessian)))
        return False, None

    # check effective degrees of freedom
    if model_dat['qdim'] >= model_dat["tau"]:
        print("-> Hessian not well conditioned: "
              "Degrees of freedom {} larger than number of observation {}."
              .format(model_dat['qdim'], model_dat["tau"]))
        return False, None

    # compute coeff covariance matrix vcm_coeff (for ci),
    # setting degree of freedom to integer qdim:
    #   the effective degree of freedom would even be smaller, decreasing resvar
    resvar = sse_hat / (model_dat["tau"] - model_dat['qdim'])
    vcm_coeff = 2 * resvar * inv_hessian
    # check if vcm_coeff is posdef by Cholesky and eigenvalues
    if not utils.isposdef(vcm_coeff):
        print("-> Hessian not well conditioned: vcm_coeff not positive-definite.")
        return False, None
    min_coeffsstd = 10**(-60) # ToDo: calibrate
    coeffsstd = diag(vcm_coeff)**(1/2)
    if any(coeffsstd < min_coeffsstd):
        print("-> Hessian not well conditioned: Inverse yields very small (<{}) coeff "
              "standard deviations, min value: {}.".format(min_coeffsstd, min(coeffsstd)))
        return False, None

    if do_print:
        print("Hessian is well conditioned.")

    return True, vcm_coeff

def check_estimate_effects(model_dat, do_print=True):
    """estimate structural model given alpha in model_dat"""

    mx_hat, my_hat, sse_hat = utils.estimate_snn(model_dat, do_print)

    ex_hat, ey_hat = utils.total_effects_alg(mx_hat, my_hat, model_dat["edx"], model_dat["edy"])
    coeffs_hat = utils.coeffvec(mx_hat, my_hat, model_dat["idx"], model_dat["idy"])

    hessian_hat = sse_hess_alg(coeffs_hat, model_dat)
    check, vcm_coeff_hat = check_hessian(sse_hat, hessian_hat, model_dat, do_print)

    return check, coeffs_hat, vcm_coeff_hat, sse_hat, mx_hat, my_hat, ex_hat, ey_hat

def estimate_alpha(model_dat):
    """estimate_alpha
    
    Set model_dat["alpha"] such that the minimal regularization parameter
    alpha gives a well conditioned Hessian. We assume that regularization
    tikh (= alpha * coeffnorm) is at most a fraction of observed y variance."""
    
    # try without regularization
    alpha_final = 0
    model_dat["alpha"] = alpha_final
    print("\nEstimation without regularization:")
    print("alpha: {:10f}".format(alpha_final))
    check, *_ = check_estimate_effects(model_dat, do_print=False)
    if check: # accept alpha if Hessian is well conditioned
        print("No regularization required.")
        model_dat["alpha"] = alpha_final
        return
    
    # with regularization
    fraction = 0.001 # ToDo: calibrate, define globally # yyy
    ymvar = trace(model_dat["ymcdat"].T @ model_dat["selwei"] @ model_dat["ymcdat"])
    coeffnorm = model_dat["coeffs_theo"].T @ model_dat["coeffs_theo"]
    alpha_start = fraction * ymvar / coeffnorm

    rel = 0.01 # ToDo: calibrate, define globally # yyy
    alpha_min = 0
    alpha_max = alpha_start * 2
    alpha = (alpha_min + alpha_max) / 2
    alpha_final = None
    print("\nEstimation of regularization parameter alpha:")
    while (alpha_max - alpha_min) / alpha > rel:
        print("alpha: {:10f}".format(alpha))
        model_dat["alpha"] = alpha
        check, *_ = check_estimate_effects(model_dat, do_print=False)
        # accept new alpha if Hessian is well conditioned
        if check is False:
            alpha_min = alpha
            if not alpha_final: # no alpha found yet
                alpha_max *= 10
        else:
            alpha_max = alpha
            alpha_final = alpha
        alpha = (alpha_min + alpha_max) / 2
    
    assert alpha_final, "No valid regularization parameter alpha found."
    print("Final alpha: {:10f}".format(alpha_final))

    # increase final alpha a bit to avoid huge standard errors
    # at the border to noninvertibility
    model_dat["alpha"] = alpha_final * 1.2 # ToDo: calibrate # yy
    
    return

def estimate_effects(model_dat):
    """nonlinear estimation of linearized structural model
    using theoretical coefficients as starting values"""
    
    # set model_dat["alpha"]
    estimate_alpha(model_dat)

    # final estimation given optimal alpha
    (check, coeffs_hat, vcm_coeff_hat, sse_hat, mx_hat, my_hat, ex_hat, ey_hat
     ) = check_estimate_effects(model_dat)
    assert check, "Hessian not well conditioned."

    hessian = utils.sse_hess(model_dat, mx_hat, my_hat)
    hessian_hat = sse_hess_alg(coeffs_hat, model_dat)
    print("\nAutomatic and algebraic Hessian allclose: {}."
          .format(allclose(hessian, hessian_hat)))

    # compute estimated coefficients, effects and standard deviations
    mx_hat_std, my_hat_std = utils.compute_coeffs_std(vcm_coeff_hat, model_dat)
    ex_hat_std, ey_hat_std = utils.total_effects_std(coeffs_hat, vcm_coeff_hat, model_dat)
    exj_hat, eyj_hat, eyx_hat, eyy_hat = utils.compute_mediation_effects(
        mx_hat, my_hat, ex_hat, ey_hat, model_dat["yvars"], model_dat["final_var"])
    (exj_hat_std, eyj_hat_std, eyx_hat_std, eyy_hat_std
     ) = utils.compute_mediation_std(ex_hat_std, ey_hat_std, eyx_hat, eyy_hat,
                                     model_dat["yvars"], model_dat["final_var"])

    estimate_dat = {
        "coeffs_hat": coeffs_hat,
        "sse_hat": sse_hat,
        "hessian_hat": hessian_hat,
        "vcm_coeff_hat": vcm_coeff_hat,
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
        bias_std = (2 * resvar * (1 / hess_i))**(1/2)
        biases_std[bias_ind] = bias_std

    return biases, biases_std

def estimate_models(model_dat):
    """estimation of modification indicators of level model"""

    # estimate linear models
    estimate_dat = estimate_effects(model_dat)

    # estimate level modification indicators, given theoretical level model
    biases, biases_std = estimate_biases(model_dat)

    # extend estimate_dat by biases
    estimate_dat["biases"] = biases
    estimate_dat["biases_std"] = biases_std

    return estimate_dat
