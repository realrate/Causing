# -*- coding: utf-8 -*-
"""Create analysis data for individual graph values."""

from collections import defaultdict

from numpy import median, zeros


def compute_delta_mat(xy_dim, m, xdat):
    """compute indiv mat"""

    if xy_dim == "x":
        dim = m.mdim
        dat = xdat
    if xy_dim == "y":
        dim = m.ndim
        dat = m.compute(xdat)

    tau = xdat.shape[1]
    dxy_mat = zeros((dim, tau))
    mat_based = zeros((dim, tau))
    for i in range(dim):
        # compute single row of dx_mat or dy_mat, subtracting median observation
        row_based = dat[i, :]
        dxy_row = row_based - median(row_based)

        dxy_mat[i, :] = dxy_row
        mat_based[i, :] = row_based

    return dxy_mat, mat_based


def make_individual_theos(m, xdat, show_nr_indiv) -> dict:
    tau = xdat.shape[1]
    all_theos = defaultdict(list)
    for obs in range(min(tau, show_nr_indiv)):
        xval = xdat[:, obs]
        theo = m.theo(xval)

        for key, val in theo.items():
            all_theos[key + "s"].append(val)

    return all_theos


def create_indiv(m, xdat, show_nr_indiv):
    """create indiv analysis data for mediation indiv graph values,
    using individual total effects and mediation effects"""

    tau = xdat.shape[1]
    indiv_theos = make_individual_theos(m, xdat, show_nr_indiv)

    # compute indiv matrices
    dx_mat, xdat_based = compute_delta_mat("x", m, xdat)
    dy_mat, yhat_based = compute_delta_mat("y", m, xdat)

    # compute direct, total and mediation indivs
    exj_indivs = zeros((m.mdim, tau))
    eyj_indivs = zeros((m.ndim, tau))
    mx_indivs = []  # tau times (ndim, mdim)
    my_indivs = []  # tau times (mdim, mdim)
    ex_indivs = []  # tau times (ndim, mdim)
    ey_indivs = []  # tau times (mdim, mdim)
    eyx_indivs = []  # tau times (ndim, mdim)
    eyy_indivs = []  # tau times (mdim, mdim)
    print()
    for obs in range(min(tau, show_nr_indiv)):
        print("Analyze individual {:5}".format(obs))
        # compute indivs row wise, using individual effects, using braodcasting for multiplication
        exj_indivs[:, obs] = indiv_theos["exj_theos"][obs] * dx_mat[:, obs].T  # (mdim)
        eyj_indivs[:, obs] = indiv_theos["eyj_theos"][obs] * dy_mat[:, obs].T  # (ndim)
        # compute mediation indivs coulumn wise, using individual eyx_theo, eyy_theos,
        # note: when multiplying a large indiv to a derivative, linear approx. errors can occur
        mx_indivs.append(indiv_theos["mx_theos"][obs] * dx_mat[:, obs])  # (ndim, mdim)
        my_indivs.append(indiv_theos["my_theos"][obs] * dy_mat[:, obs])  # (mdim, mdim)
        ex_indivs.append(indiv_theos["ex_theos"][obs] * dx_mat[:, obs])  # (ndim, mdim)
        ey_indivs.append(indiv_theos["ey_theos"][obs] * dy_mat[:, obs])  # (mdim, mdim)
        eyx_indivs.append(
            indiv_theos["eyx_theos"][obs] * dx_mat[:, obs]
        )  # (ndim, mdim)
        eyy_indivs.append(
            indiv_theos["eyy_theos"][obs] * dy_mat[:, obs]
        )  # (mdim, mdim)

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
