# -*- coding: utf-8 -*-
"""Utilities."""

# pylint: disable=invalid-name # spyder cannot read good-names from .pylintrc
# pylint: disable=E1101 # "torch has nor 'DoubleTensor' menber"

from typing import Tuple, List
from collections import defaultdict
from copy import copy, deepcopy

import pydot
import sys
from math import floor, log10

import numpy as np
from numpy.random import multivariate_normal, seed
from numpy import (
    allclose,
    array,
    concatenate,
    count_nonzero,
    diag,
    eye,
    empty,
    fill_diagonal,
    hstack,
    isnan,
    kron,
    median,
    nan,
    ones,
    reshape,
    std,
    tile,
    var,
    vstack,
    zeros,
)
import numdifftools as nd
from numpy.linalg import cholesky, inv, norm
from pandas import DataFrame
from scipy.optimize import minimize
from sympy import diff, Heaviside, lambdify
import sympy
import torch
import pathlib

from causing import svg

# set numpy random seed
seed(1002)


def nan_to_zero(x: np.array) -> np.array:
    """Replace NaNs with zeros for parts of code that can't deal with NaNs"""
    x = x.copy()
    x[x != x] = 0
    return x


def adjacency(model_dat):
    """numeric function for model and direct effects, identification matrices"""

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
                if hasattr(equationsx[j], "subs"):
                    equationsx[j] = equationsx[j].subs(
                        yvars[bias_ind], equationsx[bias_ind]
                    )

        return equationsx

    # algebraic equations containing xvars and yvars
    equations = define_equations(*xvars)

    # ToDo modules do not work, therefore replace_heaviside required # yyy
    # modules = [{'Heaviside': lambda x: np.heaviside(x, 0)}, 'sympy', 'numpy']
    # modules = [{'Heaviside': lambda x: 1 if x > 0 else 0}, 'sympy', 'numpy']
    modules = ["sympy", "numpy"]

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
                    print(
                        "i = {}, t = {}, yhat_it = {} {}".format(
                            i, t, yhat_it, type(yhat_it)
                        )
                    )
                    print(yvars[i], "=", eq)
            raise ValueError(e)

        return yhat.astype(np.float64)

    return model


def replace_heaviside(mxy, xvars, xval):
    """deal with sympy Min and Max giving Heaviside:
    Heaviside(x) = 0 if x < 0 and 1 if x > 0, but
    Heaviside(0) needs to be defined by user,
    we set Heaviside(0) to 0 because in general there is no sensitivity,
    the numpy heaviside function is lowercase and wants two arguments:
    an x value, and an x2 to decide what should happen for x==0
    https://stackoverflow.com/questions/60171926/sympy-name-heaviside-not-defined-within-lambdifygenerated
    """

    for i in range(mxy.shape[0]):
        for j in range(mxy.shape[1]):
            if hasattr(mxy[i, j], "subs"):
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

    from causing.model import Model

    xvars = model_dat["xvars"]
    yvars = model_dat["yvars"]
    equations = model_dat["define_equations"](*xvars)
    m = Model(xvars, yvars, model_dat["ymvars"], equations, model_dat["final_var"])

    # dimensions
    pdim = len(model_dat["ymvars"])
    tau = model_dat["xdat"].shape[1]

    # check
    if model_dat["ymdat"].shape[0] != pdim:
        raise ValueError(
            "Number of ymvars {} and ymdat {} not identical.".format(
                model_dat["ymdat"].shape[0], pdim
            )
        )

    # model summary
    print("Causing starting")
    print(
        "\nModel with {} endogenous and {} exogenous variables, "
        "{} direct effects and {} observations.".format(m.ndim, m.mdim, m.qdim, tau)
    )

    setup_dat = {
        # from Model class
        "m": m,
        "ndim": m.ndim,
        "mdim": m.mdim,
        "pdim": pdim,
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
        # other
        "tau": tau,
    }
    model_dat.update(setup_dat)

    # theoretical total effects at xmean and corresponding consistent ydet,
    # using closed form algebraic formula from sympy direct effects
    #   instead of automatic differentiation of model
    xmean = model_dat["xdat"].mean(axis=1)
    model_dat.update(m.theo(xmean))

    model_dat.update(
        make_individual_theos(
            m,
            model_dat["xdat"],
            tau,
            model_dat["show_nr_indiv"],
        )
    )

    # TODO: remove when Model.compute supports biases
    model_dat["old_model"] = adjacency(model_dat)

    return model_dat


def make_individual_theos(m, xdat, tau, show_nr_indiv) -> dict:
    all_theos = defaultdict(list)
    for obs in range(min(tau, show_nr_indiv)):
        xval = xdat[:, obs]
        theo = m.theo(xval)

        for key, val in theo.items():
            all_theos[key + "s"].append(val)

    return all_theos


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
    _, _, eyx, eyy = compute_mediation_effects(idx, idy, edx, edy, yvars, final_var)

    fdx = digital(eyx)
    fdy = digital(eyy)

    return fdx, fdy


def total_effects_alg(mx, my, edx, edy):
    """compute algebraic total effects given direct effects and identification matrices"""

    # dimensions
    ndim = mx.shape[0]

    # error if my is not normalized
    if sum(abs(diag(my))) > 0:
        raise ValueError(
            "No Normalization. Diagonal elements of 'my' differ from zero."
        )

    # total effects
    my_zeros = nan_to_zero(my)
    ey = inv(eye(ndim) - my_zeros)
    ex = ey @ nan_to_zero(mx)

    # set fixed null and unity effects numerically exactly to 0 and 1
    if edx is not None:
        ex[edx == 0] = float("NaN")
    if edy is not None:
        ey[edy == 0] = float("NaN")
        fill_diagonal(ey, 1)

    return ex, ey


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


def directvec_alg(mx, my):
    """algebraic direct effects vector column-wise
    from direct effects matrices and id matrices"""

    directy = my.T[~isnan(my.T)]
    directx = mx.T[~isnan(mx.T)]
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

    effects = directvec_alg(ex, ey)

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
                val = mat[i, j]
                if val != 0 and not (isinstance(val, float) and isnan(val)):
                    mat_digital[i, j] = 1

    return mat_digital


def print_output(model_dat, estimate_dat, indiv_dat, output_dir):
    """print theoretical and estimated values to output file"""

    m = model_dat["m"]
    tau = model_dat["xdat"].shape[1]

    # redirect stdout to output file
    orig_stdout = sys.stdout
    sys.stdout = open(output_dir / "logging.txt", "w")

    # model variables
    yx_vars = (model_dat["yvars"], model_dat["xvars"])
    yy_vars = (model_dat["yvars"], model_dat["yvars"])
    # xyvars = concatenate((model_dat["xvars"], model_dat["yvars"]), axis=0)

    # compute dataframe strings for printing
    if model_dat["estimate_bias"]:
        biases = concatenate(
            (
                estimate_dat["biases"].reshape(1, -1),
                estimate_dat["biases_std"].reshape(1, -1),
                (estimate_dat["biases"] / estimate_dat["biases_std"]).reshape(1, -1),
            )
        )
        biases_dfstr = DataFrame(
            biases, ("biases", "std", "t-values"), model_dat["yvars"]
        ).to_string()

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

    xmean = model_dat["xdat"].mean(axis=1)
    xmedian = median(model_dat["xdat"], axis=1)
    x_stats = vstack(
        (
            xmean.reshape(1, -1),
            xmedian.reshape(1, -1),
            std(model_dat["xdat"], axis=1).reshape(1, -1),
            ones(model_dat["mdim"]).reshape(1, -1),
        )
    )
    x_stats_dfstr = DataFrame(
        x_stats, ["xmean", "xmedian", "std", "manifest"], model_dat["xvars"]
    ).to_string()
    ydat_stats = vstack(
        (
            model_dat["ymdat"].mean(axis=1).reshape(1, -1),
            median(model_dat["ymdat"], axis=1).reshape(1, -1),
            std(model_dat["ymdat"], axis=1).reshape(1, -1),
            ones(model_dat["pdim"]).reshape(1, -1),
        )
    )
    ydat_stats_dfstr = DataFrame(
        ydat_stats, ["ymean", "ymedian", "std", "manifest"], model_dat["ymvars"]
    ).to_string()
    yhat = m.compute(model_dat["xdat"])
    ymean = yhat.mean(axis=1)
    ymedian = median(yhat, axis=1)
    ydet = model_dat["m"].compute(xmean)
    yhat_stats = vstack(
        (
            ymean.reshape(1, -1),
            ymedian.reshape(1, -1),
            ydet.reshape(1, -1),
            std(yhat, axis=1).reshape(1, -1),
            diag(model_dat["selmat"]).reshape(1, -1),
        )
    )
    yhat_stats_dfstr = DataFrame(
        yhat_stats, ["ymean", "ymedian", "ydet", "std", "manifest"], model_dat["yvars"]
    ).to_string()

    # xydat = concatenate((model_dat["xdat"], model_dat["yhat"]), axis=0)
    # xydat_dfstr = DataFrame(xydat, xyvars, range(model_dat["tau"])).to_string()

    # dx_mat_df = DataFrame(indiv_dat["dx_mat"], model_dat["xvars"], range(model_dat["tau"]))
    # dy_mat_df = DataFrame(indiv_dat["dy_mat"], model_dat["yvars"], range(model_dat["tau"]))
    # dx_mat_dfstr = dx_mat_df.to_string()
    # dy_mat_dfstr = dy_mat_df.to_string()

    # model summary
    print("Causing output file")
    print(
        "\nModel with {} endogenous and {} exogenous variables, "
        "{} direct effects and {} observations.".format(
            model_dat["ndim"], model_dat["mdim"], model_dat["qdim"], tau
        )
    )

    # alpha
    print()
    print("alpha: {:10f}, dof: {:10f}".format(model_dat["alpha"], model_dat["dof"]))

    # biases
    if model_dat["estimate_bias"]:
        print()
        print("biases:")
        print(biases_dfstr)

    # algebraic direct and total effects
    # print("\nmx_alg:")
    # print(np.array2string(model_dat["mx_alg"]))
    # print("\nmy_alg:")
    # print(np.array2string(model_dat["my_alg"]))

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
    sys.stdout.close()
    sys.stdout = orig_stdout


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
                vec_mat[:, k] = reshape(mi, ndim * mdim, order="F")
                k += 1

    return vec_mat


def render_dot(dot_str, out_type=None):
    """render Graphviz graph from dot_str to svg or other formats using pydot"""

    if out_type == "svg":
        # avoid svg UTF-8 problems for german umlauts
        dot_str = "".join([i if ord(i) < 128 else "&#%s;" % ord(i) for i in dot_str])
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
    # if graphs diretory not exist
    path = pathlib.Path(path) / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    graph.write_svg(path / f"{filename}.svg")

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


def round_sig(x, sig=2) -> float:
    """Round x to the given number of significant figures"""
    if x == 0 or isnan(x):
        return x
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def round_sig_recursive(x, sig=2):
    """Round all floats in x to the given number of significant figures

    x can be a nested data structure.
    """
    if isinstance(x, dict):
        return {key: round_sig_recursive(value, sig) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return x.__class__(round_sig_recursive(value, sig) for value in x)
    if isinstance(x, float):
        return round_sig(x, sig)

    return x
