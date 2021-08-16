# -*- coding: utf-8 -*-
"""Create direct, total and mediation Graphviz graph from dot_str using pydot."""

from typing import Dict
from itertools import chain
import numpy as np
from numpy import amax, array_equal, allclose, isnan, logical_and
from pandas import DataFrame
import json
from causing import utils
from causing.estimate import tvals
from sympy import symbols
from numpy import array as numpy_arr


def color_scheme(value, base):
    """compute colorscheme and color"""

    # default values # ToDo: set globally
    colorscheme_wo_nr = "rdylgn"
    colorscheme_nr = 9
    colorscheme_grey = "X11"
    color_grey = "snow3"

    # compute color
    if isnan(value):
        color = 0
    else:
        color_num = value * colorscheme_nr / (2 * base)
        color = max(min(int(1 + colorscheme_nr / 2 + color_num), colorscheme_nr), 1)

    # compute colorscheme
    if colorscheme_nr % 2 == 1 and color == (colorscheme_nr + 1) / 2:
        colorscheme = colorscheme_grey
        color = color_grey  # type: ignore
    else:
        colorscheme = colorscheme_wo_nr + str(colorscheme_nr)

    return colorscheme, color


def color_str(wei, base, line_colored, color, colortrans):
    """compute color string"""
    if not color:
        return ""

    if colortrans and wei:
        wei = colortrans(wei)
    colorscheme, color = color_scheme(wei, base)
    col_str = (
        f", \n                       colorscheme = {colorscheme}"
        ', style = "rounded,filled"'
        f", fillcolor  = {str(color)}"
    )
    if line_colored:
        col_str += ", penwidth = " + str(2)
        col_str += ", color = " + str(color)

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
        mat_id_df = mat_id_df.loc[
            :, logical_and((mat_id_df == 0).all(axis=0), (mat_id_df == 0).all(axis=1))
        ]
        mat_id_df = mat_id_df.loc[(mat_id_df == 0).all(axis=1), :]
        sing_nod = list(mat_id_df)

    return sing_nod


def dot(
    # main graph data
    xnodes,
    ynodes,
    weights,
    nodeff,
    # color params
    specific_color_str,
    # other
    show_in_percent: bool,
    node_name: Dict[str, str],
) -> str:
    """create inner graphviz dot_string,
    do not show edges with exact zero weight, irrespective of id matrix"""

    # single nodes
    sing_nod = single_nodes(xnodes, ynodes, weights)

    dot_str = ""
    # edges
    for row, ynode in enumerate(ynodes):
        for col, xnode in enumerate(xnodes):
            wei = weights[row, col]
            if isnan(wei) or wei == 0 or xnode == ynode:
                continue

            if show_in_percent:
                wei_str = "{}{}".format(utils.roundec(100 * wei), "%")  # perc
            else:
                wei_str = utils.roundec(wei)

            col_str = specific_color_str(wei, True)
            dot_str += '         "{}" -> "{}" [label = "{}"{}];\n'.format(
                xnode, ynode, wei_str, col_str
            )

    # nodes
    for i, xnode in enumerate(xnodes):
        if xnode in sing_nod:
            continue

        if nodeff is not None and not isnan(nodeff[i]):
            # if no nodeff given, or some elements are nan (tval ey diag)
            if show_in_percent:
                nodeff_str = "{}{}".format(utils.roundec(100 * nodeff[i]), "%")  # perc
            else:
                nodeff_str = utils.roundec(nodeff[i])
            col_str = specific_color_str(nodeff[i], False)
        else:
            nodeff_str = ""
            col_str = ""
        xnode_show = node_name.get(str(xnode), str(xnode))
        dot_str += '         "{}"[label = "{}\\n{}"{}];\n'.format(
            xnode, xnode_show, nodeff_str, col_str
        )

    return dot_str


def compute_color_base(datas):
    """compute color base over list of array with date for nodes and weights"""

    flattened_abs_values = chain(*(abs(d).flat for d in datas if d is not None))
    base = np.nanmax(list(flattened_abs_values))

    if allclose(base, 0):
        print("All values in graph close to zero. Set coloring base to one.")
        base = 1

    return base


def create_and_save_graph(
    xnodes,
    ynodes,
    x_weights_idmat_nodeff,
    y_weights_idmat_nodeff,
    color,
    dir_path,
    filename,
    final_var_is_rat_var,
    node_name,
    colortrans=None,
):
    """create graph as dot string, save it as png and return it as svg"""
    (x_weights, x_nodeff) = x_weights_idmat_nodeff
    (y_weights, y_nodeff) = y_weights_idmat_nodeff

    form = (
        "         node [style=rounded]\n"
        "         node [shape=box]\n"
        '         ratio="compress"\n'
    )

    if color is True:
        base = compute_color_base(
            [  # absolute max over all values
                x_weights,
                x_nodeff,
                y_weights,
                y_nodeff,
            ]
        )
    elif color is None:
        base = None
    else:
        base = abs(color)  # e.g. color = 2 for t-values

    def specific_color_str(wei: float, line_colored: bool) -> str:
        return color_str(wei, base, line_colored, color, colortrans)

    show_in_percent = final_var_is_rat_var and filename.startswith("IME_")
    x_dot = dot(  # type: ignore
        xnodes,
        ynodes,
        x_weights,
        x_nodeff,
        specific_color_str,
        show_in_percent,
        node_name,
    )
    y_dot = dot(  # type: ignore
        ynodes,
        ynodes,
        y_weights,
        y_nodeff,
        specific_color_str,
        show_in_percent,
        node_name,
    )
    dot_str = "digraph { \n" + form + x_dot + y_dot + "        }"

    utils.save_graph(dir_path, filename, dot_str)

    graph_svg = utils.render_dot(dot_str, out_type="svg")

    return graph_svg


def create_graphs(graph_json, output_dir, node_name, show_nr_indiv):
    """creates direct, total and mediation graph,
    for theoretical model and estimated model"""

    dir_path = output_dir
    xnodes = symbols(graph_json["xnodes"])
    ynodes = symbols(graph_json["ynodes"])
    idx = numpy_arr(graph_json["idx"])
    idy = numpy_arr(graph_json["idy"])
    final_var_is_rat_var = graph_json["final_var_is_rat_var"]

    print("\nAverage and estimated graphs")

    print("ADE")
    direct_graph = create_and_save_graph(
        xnodes,
        ynodes,
        (np.array(graph_json["mx_theo"]), None),
        (np.array(graph_json["my_theo"]), None),
        False,
        dir_path,
        "ADE",
        final_var_is_rat_var,
        node_name,
    )

    print("AME")
    # AME Params
    eyx_theo = numpy_arr(graph_json["eyx_theo"])
    eyy_theo = numpy_arr(graph_json["eyy_theo"])
    fdx = numpy_arr(graph_json["fdx"])
    fdy = numpy_arr(graph_json["fdy"])
    exj_theo = numpy_arr(graph_json["exj_theo"])
    eyj_theo = numpy_arr(graph_json["eyj_theo"])

    mediation_graph = create_and_save_graph(
        xnodes,
        ynodes,
        (eyx_theo, exj_theo),
        (eyy_theo, eyj_theo),
        False,
        dir_path,
        "AME",
        final_var_is_rat_var,
        node_name,
    )

    if "mx_hat" in graph_json:
        # If estimate has not been done, the keys necessary
        # for these graphs are missing.

        print("EDE")
        # EDE parmas
        mx_hat = numpy_arr(graph_json["mx_hat"])
        my_hat = numpy_arr(graph_json["my_hat"])
        direct_hat_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (mx_hat, None),
            (my_hat, None),
            False,
            dir_path,
            "EDE",
            final_var_is_rat_var,
            node_name,
        )

        print("ED0")
        # ED0 params
        mx_hat = numpy_arr(graph_json["mx_hat"])
        my_hat = numpy_arr(graph_json["my_hat"])
        mx_hat_std = numpy_arr(graph_json["mx_hat_std"])
        my_hat_std = numpy_arr(graph_json["my_hat_std"])

        direct_tval_graph_0 = create_and_save_graph(
            xnodes,
            ynodes,
            (tvals(mx_hat, mx_hat_std), None),
            (tvals(my_hat, my_hat_std), None),
            2,
            dir_path,
            "ED0",
            final_var_is_rat_var,
            node_name,
            lambda x: abs(x),
        )

        print("EME")
        # EME Params
        eyx_hat = numpy_arr(graph_json["eyx_hat"])
        eyy_hat = numpy_arr(graph_json["eyy_hat"])
        exj_hat = numpy_arr(graph_json["exj_hat"])
        eyj_hat = numpy_arr(graph_json["eyj_hat"])
        mediation_hat_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (eyx_hat, exj_hat),
            (eyy_hat, eyj_hat),
            False,
            dir_path,
            "EME",
            final_var_is_rat_var,
            node_name,
        )

        # EM0 Parms
        eyx_hat_std = numpy_arr(graph_json["eyx_hat_std"])
        exj_hat_std = numpy_arr(graph_json["exj_hat_std"])
        eyy_hat_std = numpy_arr(graph_json["eyy_hat_std"])
        eyj_hat_std = numpy_arr(graph_json["eyj_hat_std"])

        print("EM0")
        mediation_tval_graph_0 = create_and_save_graph(
            xnodes,
            ynodes,
            (tvals(eyx_hat, eyx_hat_std), tvals(exj_hat, exj_hat_std)),
            (tvals(eyy_hat, eyy_hat_std), tvals(eyj_hat, eyj_hat_std)),
            2,
            dir_path,
            "EM0",
            final_var_is_rat_var,
            node_name,
            lambda x: abs(x),
        )

        # ED1 json
        mx_theo = numpy_arr(graph_json["mx_theo"])
        my_theo = numpy_arr(graph_json["my_theo"])
        print("ED1")
        direct_tval_graph_1 = create_and_save_graph(
            xnodes,
            ynodes,
            ((tvals(mx_hat - mx_theo, mx_hat_std)), None),
            ((tvals(my_hat - my_theo, my_hat_std)), None),
            2,
            dir_path,
            "ED1",
            final_var_is_rat_var,
            node_name,
            lambda x: -abs(x),
        )

        print("EM1")
        # EM1 params
        mediation_tval_graph_1 = create_and_save_graph(
            xnodes,
            ynodes,
            (
                (tvals(eyx_hat - eyx_theo, eyx_hat_std)),
                (tvals(exj_hat - exj_theo, exj_hat_std)),
            ),
            (
                (tvals(eyy_hat - eyy_theo, eyy_hat_std)),
                (tvals(eyj_hat - eyj_theo, eyj_hat_std)),
            ),
            2,
            dir_path,
            "EM1",
            final_var_is_rat_var,
            node_name,
            lambda x: -abs(x),
        )

    edx = numpy_arr(graph_json["edx"])
    edy = numpy_arr(graph_json["edy"])

    if graph_json.get("is_all_graph"):
        print("ATE")
        ex_theo = numpy_arr(graph_json["ex_theo"])
        ey_theo = numpy_arr(graph_json["ey_theo"])
        total_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (ex_theo, None),
            (ey_theo, None),
            False,
            dir_path,
            "ATE",
            final_var_is_rat_var,
            node_name,
        )
        if "mx_hat" in graph_json:
            # If estimate has not been done, the keys necessary
            # for these graphs are missing.
            print("ETE")
            # ETE Params
            ex_hat = numpy_arr(graph_json["ex_hat"])
            ey_hat = numpy_arr(graph_json["ey_hat"])

            total_hat_graph = create_and_save_graph(
                xnodes,
                ynodes,
                (ex_hat, None),
                (ey_hat, None),
                False,
                dir_path,
                "ETE",
                final_var_is_rat_var,
                node_name,
            )
            print("ET0")
            ex_hat_std = numpy_arr(graph_json["ex_hat_std"])
            ey_hat_std = numpy_arr(graph_json["ey_hat_std"])
            total_tval_graph_0 = create_and_save_graph(
                xnodes,
                ynodes,
                (tvals(ex_hat, ex_hat_std), None),
                (tvals(ey_hat, ey_hat_std), None),
                2,
                dir_path,
                "ET0",
                final_var_is_rat_var,
                node_name,
                lambda x: abs(x),
            )
            print("ET1")
            total_tval_graph_1 = create_and_save_graph(
                xnodes,
                ynodes,
                ((tvals(ex_hat - ex_theo, ex_hat_std)), None),
                ((tvals(ey_hat - ey_theo, ey_hat_std)), None),
                2,
                dir_path,
                "ET1",
                final_var_is_rat_var,
                node_name,
                lambda x: -abs(x),
            )
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

    mx_indivs = [numpy_arr(i) for i in graph_json["mx_indivs"]]
    my_indivs = [numpy_arr(i) for i in graph_json["my_indivs"]]
    eyx_indivs = [numpy_arr(i) for i in graph_json["eyx_indivs"]]
    eyy_indivs = [numpy_arr(i) for i in graph_json["eyy_indivs"]]
    exj_indivs = numpy_arr(graph_json["exj_indivs"])
    eyj_indivs = numpy_arr(graph_json["eyj_indivs"])
    ex_indivs = [numpy_arr(i) for i in graph_json["ex_indivs"]]
    ey_indivs = [numpy_arr(i) for i in graph_json["ey_indivs"]]

    for i in range(show_nr_indiv):
        # compute color base for each individual separately
        # using _indiv quantities based on _theo quantities times absolute deviation from median
        print("Generate graphs for individual {:5}".format(i))

        print("IDE")
        direct_indiv_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (mx_indivs[i], None),
            (my_indivs[i], None),
            True,
            dir_path,
            "IDE" + "_" + str(i),
            final_var_is_rat_var,
            node_name,
        )
        direct_indiv_graphs.append(direct_indiv_graph)
        print("IME")
        mediation_indiv_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (eyx_indivs[i], exj_indivs[:, i]),
            (eyy_indivs[i], eyj_indivs[:, i]),
            True,
            dir_path,
            "IME" + "_" + str(i),
            final_var_is_rat_var,
            node_name,
        )
        mediation_indiv_graphs.append(mediation_indiv_graph)
        print("ITE")
        total_indiv_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (ex_indivs[i], None),
            (ey_indivs[i], None),
            True,
            dir_path,
            "ITE" + "_" + str(i),
            final_var_is_rat_var,
            node_name,
        )
        total_indiv_graphs.append(total_indiv_graph)

    # render and return graph_dat
    graph_dat = {
        # average graphs
        "direct_graph": direct_graph,
        "total_graph": total_graph,
        "mediation_graph": mediation_graph,
        # individual graphs
        "direct_indiv_graphs": direct_indiv_graphs,
        "total_indiv_graphs": total_indiv_graphs,
        "mediation_indiv_graphs": mediation_indiv_graphs,
    }

    if "mx_hat" in graph_json:
        # If estimate has not been done, the keys necessary
        # for these graphs are missing.
        graph_dat.update(
            {
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
            }
        )

    return graph_dat


def sym_to_str(sym_list):
    return ", ".join(str(i) for i in sym_list)


def create_json_graphs(m, xdat, estimate_dat, indiv_dat, mean_theo, show_nr_indiv):
    tau = xdat.shape[1]
    model_json = {
        "idx": m.idx.tolist(),
        "idy": m.idy.tolist(),
        # AME_json
        "eyx_theo": mean_theo["eyx_theo"].tolist(),  # nm_array
        "eyy_theo": mean_theo["eyy_theo"].tolist(),  # nm_array
        "fdx": m.fdx.tolist(),  # nm_array
        "fdy": m.fdy.tolist(),  # nm_array
        "exj_theo": mean_theo["exj_theo"].tolist(),  # nm_array
        "eyj_theo": mean_theo["eyj_theo"].tolist(),  # nm_array
        # ED1_json
        "mx_theo": mean_theo["mx_theo"].tolist(),
        "my_theo": mean_theo["my_theo"].tolist(),
        # EM1_json
        # ATE
        "ex_theo": mean_theo["ex_theo"].tolist(),
        "ey_theo": mean_theo["ey_theo"].tolist(),
        "edx": m.edx.tolist(),
        "edy": m.edy.tolist(),
        # indiv_dat
        "mx_indivs": [indiv.tolist() for indiv in indiv_dat["mx_indivs"]],
        "my_indivs": [indiv.tolist() for indiv in indiv_dat["my_indivs"]],
        "eyx_indivs": [indiv.tolist() for indiv in indiv_dat["eyx_indivs"]],
        "eyy_indivs": [indiv.tolist() for indiv in indiv_dat["eyy_indivs"]],
        "exj_indivs": indiv_dat["exj_indivs"].tolist(),
        "eyj_indivs": indiv_dat["eyj_indivs"].tolist(),
        "ex_indivs": [indiv.tolist() for indiv in indiv_dat["ex_indivs"]],
        "ey_indivs": [indiv.tolist() for indiv in indiv_dat["ey_indivs"]],
    }

    if estimate_dat:
        model_json.update(
            {
                # EDE json
                "mx_hat": estimate_dat["mx_hat"].tolist(),
                "my_hat": estimate_dat["my_hat"].tolist(),
                # EME json
                "eyx_hat": estimate_dat["eyx_hat"].tolist(),
                "eyy_hat": estimate_dat["eyy_hat"].tolist(),
                "exj_hat": estimate_dat["exj_hat"].tolist(),  # nm_array
                "eyj_hat": estimate_dat["eyj_hat"].tolist(),  # nm_array
                # ED0_json
                "mx_hat_std": estimate_dat["mx_hat_std"].tolist(),
                "my_hat_std": estimate_dat["my_hat"].tolist(),
                # EM0_json
                "eyx_hat_std": estimate_dat["eyx_hat_std"].tolist(),
                "exj_hat_std": estimate_dat["exj_hat_std"].tolist(),
                "eyy_hat_std": estimate_dat["eyy_hat_std"].tolist(),
                "eyj_hat_std": estimate_dat["eyj_hat_std"].tolist(),
                # ETE json
                "ex_hat": estimate_dat["ex_hat"].tolist(),
                "ey_hat": estimate_dat["ey_hat"].tolist(),
                # ET0 json
                "ex_hat_std": estimate_dat["ex_hat_std"].tolist(),
                "ey_hat_std": estimate_dat["ey_hat_std"].tolist(),
            }
        )

    return model_json
