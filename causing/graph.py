# -*- coding: utf-8 -*-
"""Create direct, total and mediation Graphviz graph from dot_str using pydot."""

from typing import Dict, Sequence, Optional
from itertools import chain
import numpy as np
from numpy import amax, array_equal, allclose, isnan, logical_and
from pandas import DataFrame
import json
from causing import utils
from causing.estimate import tvals
from causing.model import Model
from sympy import symbols


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

    xnodes_with_edge = set()
    dot_str = ""
    # edges
    for row, ynode in enumerate(ynodes):
        for col, xnode in enumerate(xnodes):
            wei = weights[row, col]
            if isnan(wei) or xnode == ynode:
                continue

            if show_in_percent:
                wei_str = "{}{}".format(utils.roundec(100 * wei), "%")  # perc
            else:
                wei_str = utils.roundec(wei)

            col_str = specific_color_str(wei, True)
            dot_str += '         "{}" -> "{}" [label = "{}"{}];\n'.format(
                xnode, ynode, wei_str, col_str
            )
            xnodes_with_edge.add(xnode)
            xnodes_with_edge.add(ynode)

    # nodes
    for i, xnode in enumerate(xnodes):
        if xnode not in xnodes_with_edge:
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
    dir_path,
    filename,
    node_name,
    color=False,
    colortrans=None,
    show_in_percent=False,
):
    """create graph as dot string, save it as png and return it as svg"""
    (x_weights, x_nodeff) = x_weights_idmat_nodeff
    (y_weights, y_nodeff) = y_weights_idmat_nodeff
    x_weights = np.array(x_weights)
    y_weights = np.array(y_weights)
    if x_nodeff is not None:
        x_nodeff = np.array(x_nodeff)
    if y_nodeff is not None:
        y_nodeff = np.array(y_nodeff)

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


def create_graphs(
    m: Model,
    graph_json,
    output_dir,
    node_name,
    show_nr_indiv,
    final_var_in_percent=False,
    ids: Optional[Sequence[str]] = None,
):
    """creates direct, total and mediation graph,
    for theoretical model and estimated model"""

    def make_graph(filename, x_weights_idmat_nodeff, y_weights_idmat_nodeff, **kwargs):
        print(filename)
        xnodes = [str(var) for var in m.xvars]
        ynodes = [str(var) for var in m.yvars]
        return create_and_save_graph(
            xnodes,
            ynodes,
            x_weights_idmat_nodeff,
            y_weights_idmat_nodeff,
            output_dir,
            filename,
            node_name,
            **kwargs,
        )

    print("\nAverage and estimated graphs")
    direct_graph = make_graph(
        "ADE",
        (np.array(graph_json["mx_theo"]), None),
        (np.array(graph_json["my_theo"]), None),
    )
    mediation_graph = make_graph(
        "AME",
        (graph_json["eyx_theo"], graph_json["exj_theo"]),
        (graph_json["eyy_theo"], graph_json["eyj_theo"]),
    )
    total_graph = make_graph(
        "ATE",
        (graph_json["ex_theo"], None),
        (graph_json["ey_theo"], None),
    )

    # mediation graphs
    direct_indiv_graphs = []
    total_indiv_graphs = []
    mediation_indiv_graphs = []
    print()

    for i in range(min(show_nr_indiv, len(graph_json["mx_indivs"]))):
        if ids:
            item_id = ids[i]
        else:
            item_id = str(i)

        # compute color base for each individual separately
        # using _indiv quantities based on _theo quantities times absolute deviation from median
        print("Generate graphs for individual {:5}".format(i))

        # IDE
        direct_indiv_graph = make_graph(
            f"IDE_{item_id}",
            (graph_json["mx_indivs"][i], None),
            (graph_json["my_indivs"][i], None),
            color=True,
        )
        direct_indiv_graphs.append(direct_indiv_graph)

        # IME
        mediation_indiv_graph = make_graph(
            f"IME_{item_id}",
            (graph_json["eyx_indivs"][i], np.array(graph_json["exj_indivs"])[:, i]),
            (graph_json["eyy_indivs"][i], np.array(graph_json["eyj_indivs"])[:, i]),
            color=True,
            show_in_percent=final_var_in_percent,
        )
        mediation_indiv_graphs.append(mediation_indiv_graph)

        # ITE
        total_indiv_graph = make_graph(
            f"ITE_{item_id}",
            (graph_json["ex_indivs"][i], None),
            (graph_json["ey_indivs"][i], None),
            color=True,
        )
        total_indiv_graphs.append(total_indiv_graph)

    return {
        # average graphs
        "direct_graph": direct_graph,
        "total_graph": total_graph,
        "mediation_graph": mediation_graph,
        # individual graphs
        "direct_indiv_graphs": direct_indiv_graphs,
        "total_indiv_graphs": total_indiv_graphs,
        "mediation_indiv_graphs": mediation_indiv_graphs,
    }


def create_estimate_graphs(
    m: Model, estimate_dat, graph_json, output_dir, node_name={}
):
    def make_graph(filename, x_weights_idmat_nodeff, y_weights_idmat_nodeff, **kwargs):
        print(filename)
        xnodes = [str(var) for var in m.xvars]
        ynodes = [str(var) for var in m.yvars]
        return create_and_save_graph(
            xnodes,
            ynodes,
            x_weights_idmat_nodeff,
            y_weights_idmat_nodeff,
            output_dir,
            filename,
            node_name,
            **kwargs,
        )

    direct_hat_graph = make_graph(
        "EDE",
        (estimate_dat["mx_hat"], None),
        (estimate_dat["my_hat"], None),
    )

    # ED0
    mx_hat = estimate_dat["mx_hat"]
    my_hat = estimate_dat["my_hat"]
    mx_hat_std = estimate_dat["mx_hat_std"]
    my_hat_std = estimate_dat["my_hat_std"]
    direct_tval_graph_0 = make_graph(
        "ED0",
        (tvals(mx_hat, mx_hat_std), None),
        (tvals(my_hat, my_hat_std), None),
        color=2,
        colortrans=lambda x: abs(x),
    )

    # EME
    eyx_hat = estimate_dat["eyx_hat"]
    eyy_hat = estimate_dat["eyy_hat"]
    exj_hat = estimate_dat["exj_hat"]
    eyj_hat = estimate_dat["eyj_hat"]
    mediation_hat_graph = make_graph(
        "EME",
        (eyx_hat, exj_hat),
        (eyy_hat, eyj_hat),
    )

    # EM0
    eyx_hat_std = estimate_dat["eyx_hat_std"]
    exj_hat_std = estimate_dat["exj_hat_std"]
    eyy_hat_std = estimate_dat["eyy_hat_std"]
    eyj_hat_std = estimate_dat["eyj_hat_std"]
    mediation_tval_graph_0 = make_graph(
        "EM0",
        (tvals(eyx_hat, eyx_hat_std), tvals(exj_hat, exj_hat_std)),
        (tvals(eyy_hat, eyy_hat_std), tvals(eyj_hat, eyj_hat_std)),
        color=2,
        colortrans=lambda x: abs(x),
    )

    # ED1
    mx_theo = np.array(graph_json["mx_theo"])
    my_theo = np.array(graph_json["my_theo"])
    direct_tval_graph_1 = make_graph(
        "ED1",
        ((tvals(mx_hat - mx_theo, mx_hat_std)), None),
        ((tvals(my_hat - my_theo, my_hat_std)), None),
        color=2,
        colortrans=lambda x: -abs(x),
    )

    # EM1
    eyx_theo = graph_json["eyx_theo"]
    eyy_theo = graph_json["eyy_theo"]
    exj_theo = graph_json["exj_theo"]
    eyj_theo = graph_json["eyj_theo"]
    mediation_tval_graph_1 = make_graph(
        "EM1",
        (
            (tvals(eyx_hat - eyx_theo, eyx_hat_std)),
            (tvals(exj_hat - exj_theo, exj_hat_std)),
        ),
        (
            (tvals(eyy_hat - eyy_theo, eyy_hat_std)),
            (tvals(eyj_hat - eyj_theo, eyj_hat_std)),
        ),
        color=2,
        colortrans=lambda x: -abs(x),
    )

    # ETE
    ex_hat = np.array(estimate_dat["ex_hat"])
    ey_hat = np.array(estimate_dat["ey_hat"])
    total_hat_graph = make_graph(
        "ETE",
        (ex_hat, None),
        (ey_hat, None),
    )
    # ET0
    ex_hat_std = np.array(estimate_dat["ex_hat_std"])
    ey_hat_std = np.array(estimate_dat["ey_hat_std"])
    total_tval_graph_0 = make_graph(
        "ET0",
        (tvals(ex_hat, ex_hat_std), None),
        (tvals(ey_hat, ey_hat_std), None),
        color=2,
        colortrans=lambda x: abs(x),
    )
    # ET1
    ex_theo = np.array(graph_json["ex_theo"])
    ey_theo = np.array(graph_json["ey_theo"])
    total_tval_graph_1 = make_graph(
        "ET1",
        ((tvals(ex_hat - ex_theo, ex_hat_std)), None),
        ((tvals(ey_hat - ey_theo, ey_hat_std)), None),
        color=2,
        colortrans=lambda x: -abs(x),
    )

    return {
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


def create_json_graphs(m, xdat, indiv_dat, mean_theo, show_nr_indiv):
    model_json = {
        # AME_json
        "eyx_theo": mean_theo["eyx_theo"],
        "eyy_theo": mean_theo["eyy_theo"],
        "exj_theo": mean_theo["exj_theo"],
        "eyj_theo": mean_theo["eyj_theo"],
        # ED1_json
        "mx_theo": mean_theo["mx_theo"],
        "my_theo": mean_theo["my_theo"],
        # EM1_json
        # ATE
        "ex_theo": mean_theo["ex_theo"],
        "ey_theo": mean_theo["ey_theo"],
        # indiv_dat
        "mx_indivs": indiv_dat["mx_indivs"],
        "my_indivs": indiv_dat["my_indivs"],
        "eyx_indivs": indiv_dat["eyx_indivs"],
        "eyy_indivs": indiv_dat["eyy_indivs"],
        "exj_indivs": indiv_dat["exj_indivs"],
        "eyj_indivs": indiv_dat["eyj_indivs"],
        "ex_indivs": indiv_dat["ex_indivs"],
        "ey_indivs": indiv_dat["ey_indivs"],
    }

    return model_json
