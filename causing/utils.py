# -*- coding: utf-8 -*-
"""Utilities."""

# pylint: disable=invalid-name # spyder cannot read good-names from .pylintrc
# pylint: disable=E1101 # "torch has nor 'DoubleTensor' menber"

from typing import IO, Sequence
from copy import deepcopy
import json

import pydot
from math import floor, log10

import numpy as np
from numpy.random import seed
from numpy import (
    concatenate,
    count_nonzero,
    diag,
    eye,
    fill_diagonal,
    isnan,
    median,
    ones,
    reshape,
    std,
    vstack,
    zeros,
)
from numpy.linalg import inv, norm
from pandas import DataFrame
import sympy
import torch
import pathlib

# set numpy random seed
seed(1002)


def nan_to_zero(x: np.array) -> np.array:
    """Replace NaNs with zeros for parts of code that can't deal with NaNs"""
    x = x.copy()
    x[x != x] = 0
    return x


@np.vectorize
def replace_heaviside(formula):
    """Set Heaviside(0) = 0

    Differentiating sympy Min and Max is giving Heaviside:
    Heaviside(x) = 0 if x < 0 and 1 if x > 0, but
    Heaviside(0) needs to be defined by user.

    We set Heaviside(0) to 0 because in general there is no sensitivity.  This
    done by setting the second argument to zero.
    """
    if not isinstance(formula, sympy.Expr):
        return formula
    w = sympy.Wild("w")
    return formula.replace(sympy.Heaviside(w), sympy.Heaviside(w, 0))


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


def print_output(
    m, xdat, estimate_dat, estimate_input, indiv_dat, mean_theo, output_file: IO
):
    """print theoretical and estimated values to output file"""

    tau = xdat.shape[1]

    # compute dataframe strings for printing
    if estimate_input["estimate_bias"]:
        biases = concatenate(
            (
                estimate_dat["biases"].reshape(1, -1),
                estimate_dat["biases_std"].reshape(1, -1),
                (estimate_dat["biases"] / estimate_dat["biases_std"]).reshape(1, -1),
            )
        )
        biases_dfstr = DataFrame(
            biases, ("biases", "std", "t-values"), m.yvars
        ).to_string()

    hessian_hat_dfstr = DataFrame(estimate_dat["hessian_hat"]).to_string()

    yhat = m.compute(xdat)
    ymean = yhat.mean(axis=1)
    ymedian = median(yhat, axis=1)
    xmean = xdat.mean(axis=1)
    ydet = m.compute(np.vstack(xmean))
    yhat_stats = vstack(
        (
            ymean.reshape(1, -1),
            ymedian.reshape(1, -1),
            ydet.reshape(1, -1),
            std(yhat, axis=1).reshape(1, -1),
        )
    )
    yhat_stats_dfstr = DataFrame(
        yhat_stats, ["ymean", "ymedian", "ydet", "std"], m.yvars
    ).to_string()

    # xydat = concatenate((model_dat["xdat"], model_dat["yhat"]), axis=0)
    # xydat_dfstr = DataFrame(xydat, xyvars, range(model_dat["tau"])).to_string()

    # dx_mat_df = DataFrame(indiv_dat["dx_mat"], model_dat["xvars"], range(model_dat["tau"]))
    # dy_mat_df = DataFrame(indiv_dat["dy_mat"], model_dat["yvars"], range(model_dat["tau"]))
    # dx_mat_dfstr = dx_mat_df.to_string()
    # dy_mat_dfstr = dy_mat_df.to_string()

    def pr(*args, **kwargs):
        print(*args, **kwargs, file=output_file)

    # model summary
    pr("Causing output file")
    pr(
        "\nModel with {} endogenous and {} exogenous variables, "
        "{} direct effects and {} observations.".format(m.ndim, m.mdim, m.qdim, tau)
    )

    # alpha
    pr()
    pr(
        "alpha: {:10f}, dof: {:10f}".format(
            estimate_input["alpha"], estimate_input["dof"]
        )
    )

    # biases
    if estimate_input["estimate_bias"]:
        pr()
        pr("biases:")
        pr(biases_dfstr)

    # algebraic direct and total effects
    # pr("\nmx_alg:")
    # pr(np.array2string(model_dat["mx_alg"]))
    # pr("\nmy_alg:")
    # pr(np.array2string(model_dat["my_alg"]))

    # descriptive statistics
    pr()
    pr("xdat:")
    pr(DataFrame(xdat.T, columns=m.xvars).describe())
    pr("ymdat:")
    pr(
        DataFrame(
            estimate_input["ymdat"].T, columns=estimate_input["ymvars"]
        ).describe()
    )
    pr("yhat:")
    pr(yhat_stats_dfstr)

    # input and output data
    # pr()
    # pr("xdat, yhat:")
    # pr(xydat_dfstr)

    def print_effects(label, prefix, columns: Sequence["str"]):
        for inner_label, array in [
            (f"{label} effects {prefix}_theo:", mean_theo[prefix + "_theo"]),
            (f"{label} effects {prefix}_hat:", estimate_dat[prefix + "_hat"]),
            (f"{label} effects {prefix}_hat_std:", estimate_dat[prefix + "_hat_std"]),
        ]:
            pr(inner_label)
            pr(DataFrame(array, index=m.yvars, columns=columns))
            pr(array.shape)

    print_effects("Exogeneous direct", "mx", m.xvars)
    print_effects("Endogeneous direct", "my", m.yvars)
    print_effects("Exogeneous total", "ex", m.xvars)
    print_effects("Endogeneous total", "ey", m.yvars)
    print_effects("Exogeneous mediation", "eyx", m.xvars)
    print_effects("Endogeneous mediation", "eyy", m.yvars)

    # hessian
    pr("\nAlgebraic Hessian at estimated direct effects hessian_hat:")
    pr(hessian_hat_dfstr)
    pr(estimate_dat["hessian_hat"].shape)

    # indiv matrices
    # pr("\nExogeneous indiv matrix dx_mat:")
    # pr(dx_mat_dfstr)
    # pr((model_dat["mdim"], model_dat["tau"]))
    # pr("\nEndogeneous indiv matrix dy_mat:")
    # pr(dy_mat_dfstr)
    # pr((model_dat["ndim"], model_dat["tau"]))


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


def render_dot(dot_str, filename):
    """render Graphviz graph from dot_str to svg using pydot"""
    pydot.graph_from_dot_data(dot_str)[0].write_svg(filename)


def save_graph(path, filename, graph_dot):
    """save graph to file as dot string and png"""

    path = pathlib.Path(path) / "graphs"
    path.mkdir(parents=True, exist_ok=True)

    # with open(path / (filename + ".txt"), "w") as file:
    #     file.write(graph_dot)

    render_dot(graph_dot, path / f"{filename}.svg")

    return


def acc(n1, n2):
    """accuracy: similarity of two numeric matrices,
    between zero (bad) and one (good)"""

    n1 = np.array(n1)
    n2 = np.array(n2)
    if norm(n1 - n2) != 0 and norm(n1 + n2) == 0:
        accuracy = 0
    elif norm(n1 - n2) == 0 and norm(n1 + n2) == 0:
        accuracy = 1
    else:
        accuracy = 1 - norm(n1 - n2) / norm(n1 + n2)

    return accuracy


@np.vectorize
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
    if isinstance(x, (float, np.ndarray)):
        return round_sig(x, sig)
    if isinstance(x, torch.Tensor):
        return x.apply_(lambda x: round_sig(x, sig))

    return x


class MatrixEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_json(data, filename, round_sig=None):
    with open(filename, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4, cls=MatrixEncoder)
