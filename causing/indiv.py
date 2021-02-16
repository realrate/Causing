# -*- coding: utf-8 -*-
"""Create analysis data for individual graph values."""

from numpy import median, zeros


def compute_indiv_row(i, xy_dim, model_dat):
    """compute single row of dx_mat or dy_mat, subtracting median observation,
    if x_basevars, y_basevars are given, data are scaled by their base_var"""

    # dat, basevars
    if xy_dim == "x":
        dat = model_dat["xdat"]
        if "base_var" in model_dat:
            basevars = model_dat["x_basevars"]
        else:
            basevars = [None] * len(model_dat["xvars"])
    if xy_dim == "y":
        dat = model_dat["yhat"]
        if "base_var" in model_dat:
            basevars = model_dat["y_basevars"]
        else:
            basevars = [None] * len(model_dat["yvars"])

    # compute dxy_row
    if basevars[i] is not None:
        # basevar: divide by basevar, subtract median, multiply by basevar
        basevar = basevars[i]
        if basevar in model_dat["xvars"]:
            ind = model_dat["xvars"].index(basevar)
            base_dat = model_dat["xdat"][ind, :]
        if basevar in model_dat["yvars"]:
            ind = model_dat["yvars"].index(basevar)
            base_dat = model_dat["yhat"][ind, :]
        row_based = dat[i, :] / base_dat
        dxy_row = (row_based - median(row_based)) * base_dat
    else:
        # None base: subtract median
        row_based = dat[i, :]
        dxy_row = row_based - median(row_based)

    return dxy_row, row_based


def compute_delta_mat(xy_dim, model_dat):
    """compute indiv mat"""

    if xy_dim == "x":
        dim = model_dat["mdim"]
    if xy_dim == "y":
        dim = model_dat["ndim"]

    dxy_mat = zeros((dim, model_dat["tau"]))
    mat_based = zeros((dim, model_dat["tau"]))
    for i in range(dim):
        dxy_row, row_based = compute_indiv_row(i, xy_dim, model_dat)
        dxy_mat[i, :] = dxy_row
        mat_based[i, :] = row_based

    return dxy_mat, mat_based


def create_indiv(model_dat):
    """create indiv analysis data for mediation indiv graph values,
    using individual total effects and mediation effects"""

    # compute indiv matrices
    dx_mat, xdat_based = compute_delta_mat("x", model_dat)
    dy_mat, yhat_based = compute_delta_mat("y", model_dat)

    # compute direct, total and mediation indivs
    exj_indivs = zeros((model_dat["mdim"], model_dat["tau"]))
    eyj_indivs = zeros((model_dat["ndim"], model_dat["tau"]))
    mx_indivs = []  # tau times (ndim, mdim)
    my_indivs = []  # tau times (mdim, mdim)
    ex_indivs = []  # tau times (ndim, mdim)
    ey_indivs = []  # tau times (mdim, mdim)
    eyx_indivs = []  # tau times (ndim, mdim)
    eyy_indivs = []  # tau times (mdim, mdim)
    print()
    for obs in range(min(model_dat["tau"],
                         model_dat["show_nr_indiv"])):
        print("Analyze individual {:5}".format(obs))
        # compute indivs row wise, using individual effects, using braodcasting for multiplication
        exj_indivs[:, obs] = model_dat["exj_theos"][obs] * dx_mat[:, obs].T  # (mdim)
        eyj_indivs[:, obs] = model_dat["eyj_theos"][obs] * dy_mat[:, obs].T  # (ndim)
        # compute mediation indivs coulumn wise, using individual eyx_theo, eyy_theos,
        # note: when multiplying a large indiv to a derivative, linear approx. errors can occur       
        mx_indivs.append(model_dat["mx_theos"][obs] * dx_mat[:, obs])  # (ndim, mdim)
        my_indivs.append(model_dat["my_theos"][obs] * dy_mat[:, obs])  # (mdim, mdim)
        ex_indivs.append(model_dat["ex_theos"][obs] * dx_mat[:, obs])  # (ndim, mdim)
        ey_indivs.append(model_dat["ey_theos"][obs] * dy_mat[:, obs])  # (mdim, mdim)
        eyx_indivs.append(model_dat["eyx_theos"][obs] * dx_mat[:, obs])  # (ndim, mdim)
        eyy_indivs.append(model_dat["eyy_theos"][obs] * dy_mat[:, obs])  # (mdim, mdim)

    indiv_dat = {
        "dx_mat": dx_mat,
        "dy_mat": dy_mat,
        "xdat_based": xdat_based,
        "yhat_based": yhat_based,
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
