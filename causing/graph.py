# -*- coding: utf-8 -*-
"""Create direct, total and mediation Graphviz graph from dot_str using pydot."""

from numpy import amax, array_equal, allclose, isnan, logical_and
from pandas import DataFrame

from causing import utils


def color_scheme(value, base):
    """compute colorscheme and color"""

    # default values # ToDo: set globally
    colorscheme_wo_nr = "rdylgn"
    colorscheme_nr = 9
    colorscheme_grey = "X11"
    color_grey = 'snow3'

    # compute color
    if isnan(value):
        color = 0
    else:
        color_num = value * colorscheme_nr / (2 * base)
        color = max(min(int(1 + colorscheme_nr / 2 + color_num), colorscheme_nr), 1)

    # compute colorscheme
    if colorscheme_nr % 2 == 1 and color == (colorscheme_nr + 1) / 2:
        colorscheme = colorscheme_grey
        color = color_grey
    else:
        colorscheme = colorscheme_wo_nr + str(colorscheme_nr)

    return colorscheme, color


def color_str(wei, base, line_colored, color, colortrans):
    """compute color string"""

    if color:
        if colortrans and wei:
            wei = colortrans(wei)
        colorscheme, color = color_scheme(wei, base)
        col_str = (', \n                       colorscheme = ' + colorscheme
                   + ', style = "rounded,filled"' + ', fillcolor  = ' + str(color))
        if line_colored:
            col_str += ', penwidth = ' + str(2)
            col_str += ', color = ' + str(color)
    else:
        col_str = ""

    return col_str


def single_nodes(xnodes, ynodes, mat_id):
    """find single nodes, without incoming or outgoing edges,
    given either id, ed or fd identification matrix"""

    mat_id_df = DataFrame(mat_id, ynodes, xnodes)

    if not array_equal(xnodes, ynodes):
        # x nodes
        mat_id_df = mat_id_df.loc[:, (mat_id_df == 0).all(axis=0)]
        sing_nod = list(mat_id_df)
    else:
        # y nodes
        mat_id_df = mat_id_df.loc[:, logical_and((mat_id_df == 0).all(axis=0),
                                                 (mat_id_df == 0).all(axis=1))]
        mat_id_df = mat_id_df.loc[(mat_id_df == 0).all(axis=1), :]
        sing_nod = list(mat_id_df)

    return sing_nod


def dot(xnodes, ynodes, weights, id_mat, nodeff,
        color, base, colortrans, filename, model_dat):
    """create inner graphviz dot_string,
    do not show edges with exact zero weight, irrespective of id matrix"""

    xdim = len(xnodes)
    ydim = len(ynodes)

    # if no id matrix is given, take zero weights as null restriction
    if id_mat is None:
        id_mat = weights

    # single nodes
    sing_nod = single_nodes(xnodes, ynodes, id_mat)

    dot_str = ""
    # edges
    for row in range(ydim):
        ynode = ynodes[row]
        for col in range(xdim):
            xnode = xnodes[col]
            wei = weights[row, col]
            if id_mat[row, col] != 0 and xnode != ynode:
                if ("base_var" in model_dat and
                        model_dat["final_var"] in model_dat["rat_var"] and
                        filename.startswith('IME_')):
                    wei_str = "{}{}".format(utils.roundec(100 * wei), "%")  # perc
                else:
                    wei_str = utils.roundec(wei)
                col_str = color_str(wei, base, True, color, colortrans)
                dot_str += ('         "{}" -> "{}" [label = "{}"{}];\n'
                            .format(xnode, ynode, wei_str, col_str))

    # nodes
    for i in range(xdim):
        xnode = xnodes[i]
        if xnode not in sing_nod:
            if nodeff is not None and not isnan(nodeff[i]):
                # if no nodeff given, or some elements are nan (tval ey diag)
                if ("base_var" in model_dat and
                        model_dat["final_var"] in model_dat["rat_var"] and
                        filename.startswith('IME_')):
                    nodeff_str = "{}{}".format(utils.roundec(100 * nodeff[i]), "%")  # perc
                else:
                    nodeff_str = utils.roundec(nodeff[i])
                col_str = color_str(nodeff[i], base, False, color, colortrans)
            else:
                nodeff_str = ""
                col_str = ""
            if "base_var" in model_dat:  # ToDo: if full_name # yyyy
                xnode_show = xnode
                # from apps.life import rr_report
                # xnode_show = rr_report.full_name(xnode)
                # xnode_show = rr_report.break_string(xnode_show)
            else:
                xnode_show = xnode
            dot_str += ('         "{}"[label = "{}\\n{}"{}];\n'
                        .format(xnode, xnode_show, nodeff_str, col_str))

    return dot_str


def compute_color_base(datas):
    """compute color base over list of array with date for nodes and weights"""

    datas = [data for data in datas if data is not None]

    maxs = []
    for data in datas:
        maxs.append(amax(abs(data)))
    base = max(maxs)

    if allclose(base, 0):
        print("All values in graph close to zero. Set coloring base to one.")
        base = 1

    return base


def create_and_save_graph(xnodes, ynodes, x_weights_idmat_nodeff, y_weights_idmat_nodeff,
                          color, dir_path, filename, model_dat, colortrans=None):
    """create graph as dot string, save it as png and return it as svg"""

    form = ("         node [style=rounded]\n"
            "         node [shape=box]\n"
            '         ratio="compress"\n')

    if color is True:
        base = compute_color_base([  # absolute max over all values
            x_weights_idmat_nodeff[0], x_weights_idmat_nodeff[2],
            y_weights_idmat_nodeff[0], y_weights_idmat_nodeff[2]])
    elif color is None:
        base = None
    else:
        base = abs(color)  # e.g. color = 2 for t-values

    x_dot = dot(xnodes, ynodes, *x_weights_idmat_nodeff,
                color, base, colortrans, filename, model_dat)
    y_dot = dot(ynodes, ynodes, *y_weights_idmat_nodeff,
                color, base, colortrans, filename, model_dat)
    dot_str = "digraph { \n" + form + x_dot + y_dot + "        }"

    utils.save_graph(dir_path, filename, dot_str)

    graph_svg = utils.render_dot(dot_str, out_type="svg")

    return graph_svg


def create_graphs(model_dat, estimate_dat, indiv_dat):
    """creates direct, total and mediation graph,
    for theoretical model and estimated model"""

    dir_path = model_dat["dir_path"]
    xnodes = model_dat["xvars"]
    ynodes = model_dat["yvars"]

    mx_theo, my_theo = utils.directmat_alg(
        model_dat["direct_theo"], model_dat["idx"], model_dat["idy"])

    print("\nAverage and estimated graphs")
    print("ADE")
    direct_graph = create_and_save_graph(
        xnodes, ynodes,
        (mx_theo, model_dat["idx"], None),
        (my_theo, model_dat["idy"], None),
        False, dir_path, "ADE", model_dat)
    print("AME")
    mediation_graph = create_and_save_graph(
        xnodes, ynodes,
        (model_dat["eyx_theo"], model_dat["fdx"], model_dat["exj_theo"]),
        (model_dat["eyy_theo"], model_dat["fdy"], model_dat["eyj_theo"]),
        False, dir_path, "AME", model_dat)

    print("EDE")
    direct_hat_graph = create_and_save_graph(
        xnodes, ynodes,
        (estimate_dat["mx_hat"], model_dat["idx"], None),
        (estimate_dat["my_hat"], model_dat["idy"], None),
        False, dir_path, "EDE", model_dat)
    print("EME")
    mediation_hat_graph = create_and_save_graph(
        xnodes, ynodes,
        (estimate_dat["eyx_hat"], model_dat["fdx"], estimate_dat["exj_hat"]),
        (estimate_dat["eyy_hat"], model_dat["fdy"], estimate_dat["eyj_hat"]),
        False, dir_path, "EME", model_dat)

    print("ED0")
    direct_tval_graph_0 = create_and_save_graph(
        xnodes, ynodes,
        (utils.tvals(estimate_dat["mx_hat"], estimate_dat["mx_hat_std"]),
         model_dat["idx"],
         None),
        (utils.tvals(estimate_dat["my_hat"], estimate_dat["my_hat_std"]),
         model_dat["idy"],
         None),
        2, dir_path, "ED0", model_dat, lambda x: abs(x))
    print("EM0")
    mediation_tval_graph_0 = create_and_save_graph(
        xnodes, ynodes,
        (utils.tvals(estimate_dat["eyx_hat"], estimate_dat["eyx_hat_std"]),
         model_dat["fdx"],
         utils.tvals(estimate_dat["exj_hat"], estimate_dat["exj_hat_std"])),
        (utils.tvals(estimate_dat["eyy_hat"], estimate_dat["eyy_hat_std"]),
         model_dat["fdy"],
         utils.tvals(estimate_dat["eyj_hat"], estimate_dat["eyj_hat_std"])),
        2, dir_path, "EM0", model_dat, lambda x: abs(x))

    print("ED1")
    direct_tval_graph_1 = create_and_save_graph(
        xnodes, ynodes,
        ((utils.tvals(estimate_dat["mx_hat"] - model_dat["mx_theo"],
                      estimate_dat["mx_hat_std"])),
         model_dat["idx"],
         None),
        ((utils.tvals(estimate_dat["my_hat"] - model_dat["my_theo"],
                      estimate_dat["my_hat_std"])),
         model_dat["idy"],
         None),
        2, dir_path, "ED1", model_dat, lambda x: -abs(x))
    print("EM1")
    mediation_tval_graph_1 = create_and_save_graph(
        xnodes, ynodes,
        ((utils.tvals(estimate_dat["eyx_hat"] - model_dat["eyx_theo"],
                      estimate_dat["eyx_hat_std"])),
         model_dat["fdx"],
         (utils.tvals(estimate_dat["exj_hat"] - model_dat["exj_theo"],
                      estimate_dat["exj_hat_std"]))),
        ((utils.tvals(estimate_dat["eyy_hat"] - model_dat["eyy_theo"],
                      estimate_dat["eyy_hat_std"])),
         model_dat["fdy"],
         (utils.tvals(estimate_dat["eyj_hat"] - model_dat["eyj_theo"],
                      estimate_dat["eyj_hat_std"]))),
        2, dir_path, "EM1", model_dat, lambda x: -abs(x))

    # show total graphs only for smaller ndim
    show_total_ndim = 10  # ToDo: set globally # yyy
    if model_dat["ndim"] < show_total_ndim:
        print("ATE")
        total_graph = create_and_save_graph(
            xnodes, ynodes,
            (model_dat["ex_theo"], model_dat["edx"], None),
            (model_dat["ey_theo"], model_dat["edy"], None),
            False, dir_path, "ATE", model_dat)
        print("ETE")
        total_hat_graph = create_and_save_graph(
            xnodes, ynodes,
            (estimate_dat["ex_hat"], model_dat["edx"], None),
            (estimate_dat["ey_hat"], model_dat["edy"], None),
            False, dir_path, "ETE", model_dat)
        print("ET0")
        total_tval_graph_0 = create_and_save_graph(
            xnodes, ynodes,
            (utils.tvals(estimate_dat["ex_hat"], estimate_dat["ex_hat_std"]),
             model_dat["edx"],
             None),
            (utils.tvals(estimate_dat["ey_hat"], estimate_dat["ey_hat_std"]),
             model_dat["edy"],
             None),
            2, dir_path, "ET0", model_dat, lambda x: abs(x))
        print("ET1")
        total_tval_graph_1 = create_and_save_graph(
            xnodes, ynodes,
            ((utils.tvals(estimate_dat["ex_hat"] - model_dat["ex_theo"],
                          estimate_dat["ex_hat_std"])),
             model_dat["edx"],
             None),
            ((utils.tvals(estimate_dat["ey_hat"] - model_dat["ey_theo"],
                          estimate_dat["ey_hat_std"])),
             model_dat["edy"],
             None),
            2, dir_path, "ET1", model_dat, lambda x: -abs(x))
    else:
        total_graph = None
        total_hat_graph = None
        total_tval_graph_0 = None
        total_tval_graph_1 = None

    # mediation graphs
    direct_indiv_graphs = []
    total_indiv_graphs = []
    mediation_indiv_graphs = []
    print()
    for i in range(min(model_dat["tau"],
                       model_dat["show_nr_indiv"])):
        # compute color base for each individual separately
        # using _indiv quantities based on _theo quantities times absolute deviation from median
        print("Generate graphs for individual {:5}".format(i))

        print("IDE")
        direct_indiv_graph = create_and_save_graph(
            xnodes, ynodes,
            (indiv_dat["mx_indivs"][i], model_dat["idx"], None),
            (indiv_dat["my_indivs"][i], model_dat["idy"], None),
            True, dir_path, "IDE" + "_" + str(i), model_dat)
        direct_indiv_graphs.append(direct_indiv_graph)
        print("IME")
        mediation_indiv_graph = create_and_save_graph(
            xnodes, ynodes,
            (indiv_dat["eyx_indivs"][i], model_dat["fdx"], indiv_dat["exj_indivs"][:, i]),
            (indiv_dat["eyy_indivs"][i], model_dat["fdy"], indiv_dat["eyj_indivs"][:, i]),
            True, dir_path, "IME" + "_" + str(i), model_dat)
        mediation_indiv_graphs.append(mediation_indiv_graph)
        if model_dat["ndim"] < show_total_ndim:
            print("ITE")
            total_indiv_graph = create_and_save_graph(
                xnodes, ynodes,
                (indiv_dat["ex_indivs"][i], model_dat["edx"], None),
                (indiv_dat["ey_indivs"][i], model_dat["edy"], None),
                True, dir_path, "ITE" + "_" + str(i), model_dat)
            total_indiv_graphs.append(total_indiv_graph)
        else:
            total_indiv_graphs.append(None)

    # render and return graph_dat
    graph_dat = {
        # average graphs
        "direct_graph": direct_graph,
        "total_graph": total_graph,
        "mediation_graph": mediation_graph,
        # estimated graphs
        "direct_hat_graph": direct_hat_graph,
        "total_hat_graph": total_hat_graph,
        "mediation_hat_graph": mediation_hat_graph,
        # tvalues graphs wrt 0
        "direct_tval_graph_0": direct_tval_graph_0,
        "total_tval_graph_0": total_tval_graph_0,
        "mediation_tval_graph_0": mediation_tval_graph_0,
        # tvalues graphs wrt 1
        "direct_tval_graph_1": direct_tval_graph_1,
        "total_tval_graph_1": total_tval_graph_1,
        "mediation_tval_graph_1": mediation_tval_graph_1,
        # individual graphs
        "direct_indiv_graphs": direct_indiv_graphs,
        "total_indiv_graphs": total_indiv_graphs,
        "mediation_indiv_graphs": mediation_indiv_graphs,
    }

    return graph_dat
