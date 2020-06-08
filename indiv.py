# -*- coding: utf-8 -*-
"""Create analysis data for individual graph values."""

from numpy import median, zeros


def compute_indiv_row(i, xy_dim, model_dat):
    """compute single non-zero row of dx_mat or dy_mat, using median observation"""

    row_nr = i
    if xy_dim == "x":
        dat = model_dat["xdat"]
    if xy_dim == "y":
        dat = model_dat["yhat"]

    indiv_row = dat[row_nr, :]
    indiv_row_delta = indiv_row - median(indiv_row)

    return indiv_row_delta, indiv_row

def compute_indiv_mat(xy_dim, model_dat):
    """compute indiv mat"""

    if xy_dim == "x":
        dim = model_dat["mdim"]
    if xy_dim == "y":
        dim = model_dat["ndim"]

    indiv_mat = zeros((dim, model_dat["tau"]))
    rel_mat = zeros((dim, model_dat["tau"]))
    for i in range(dim):
        indiv_row_delta, indiv_row = compute_indiv_row(i, xy_dim, model_dat)
        indiv_mat[i, :] = indiv_row_delta
        rel_mat[i, :] = indiv_row

    return indiv_mat, rel_mat

def create_indiv(model_dat):
    """create indiv analysis data for mediation indiv graph values,
    using individual total effects and mediation effects"""

    # compute indiv matrices
    dx_mat, relx_mat = compute_indiv_mat("x", model_dat)
    dy_mat, rely_mat = compute_indiv_mat("y", model_dat)

    # compute indivs and mediation indivs
    exj_indivs = zeros((model_dat["mdim"], model_dat["tau"]))
    eyj_indivs = zeros((model_dat["ndim"], model_dat["tau"]))
    mx_indivs = []                                                          # tau times (ndim, mdim)
    my_indivs = []                                                          # tau times (mdim, mdim)
    ex_indivs = []                                                          # tau times (ndim, mdim)
    ey_indivs = []                                                          # tau times (mdim, mdim)
    eyx_indivs = []                                                         # tau times (ndim, mdim)
    eyy_indivs = []                                                         # tau times (mdim, mdim)
    print()
    for obs in range(model_dat["show_nr_indiv"]):
        print("Analyze individual {:5}", obs)
        # compute indivs row wise, using individual effects, using braodcasting for multiplication
        exj_indivs[:, obs] = model_dat["exj_theos"][obs] * dx_mat[:, obs].T # (mdim)
        eyj_indivs[:, obs] = model_dat["eyj_theos"][obs] * dy_mat[:, obs].T # (ndim)
        # compute mediation indivs coulumn wise, using individual eyx_theo, eyy_theos,
        # note: when multiplying a large indiv to a derivative, linear approx. errors can occur
        mx_indiv = model_dat["mx_theos"][obs] * dx_mat[:, obs]              # (ndim, mdim)
        my_indiv = model_dat["my_theos"][obs] * dy_mat[:, obs]              # (mdim, mdim)
        ex_indiv = model_dat["ex_theos"][obs] * dx_mat[:, obs]              # (ndim, mdim)
        ey_indiv = model_dat["ey_theos"][obs] * dy_mat[:, obs]              # (mdim, mdim)
        eyx_indiv = model_dat["eyx_theos"][obs] * dx_mat[:, obs]            # (ndim, mdim)
        eyy_indiv = model_dat["eyy_theos"][obs] * dy_mat[:, obs]            # (mdim, mdim)
        mx_indivs.append(mx_indiv)
        my_indivs.append(my_indiv)
        ex_indivs.append(ex_indiv)
        ey_indivs.append(ey_indiv)
        eyx_indivs.append(eyx_indiv)
        eyy_indivs.append(eyy_indiv)

    indiv_dat = {
        "dx_mat": dx_mat,
        "dy_mat": dy_mat,
        "relx_mat": relx_mat,
        "rely_mat": rely_mat,
        "exj_indivs": exj_indivs,
        "eyj_indivs": eyj_indivs,
        "mx_indivs": mx_indivs,
        "my_indivs": my_indivs,
        "ex_indivs": ex_indivs,
        "ey_indivs": ey_indivs,
        "eyx_indivs": eyx_indivs,
        "eyy_indivs": eyy_indivs,
        }

    return indiv_dat
