# -*- coding: utf-8 -*-
"""Create direct, total and mediation Graphviz graph from dot_str using pydot."""

from numpy import amax, array_equal, allclose, isnan, logical_and
from pandas import DataFrame
import json
from causing import utils
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

    if color:
        if colortrans and wei:
            wei = colortrans(wei)
        colorscheme, color = color_scheme(wei, base)
        col_str = (
                ", \n                       colorscheme = "
                + colorscheme
                + ', style = "rounded,filled"'
                + ", fillcolor  = "
                + str(color)
        )
        if line_colored:
            col_str += ", penwidth = " + str(2)
            col_str += ", color = " + str(color)
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
        mat_id_df = mat_id_df.loc[
                    :, logical_and((mat_id_df == 0).all(axis=0), (mat_id_df == 0).all(axis=1))
                    ]
        mat_id_df = mat_id_df.loc[(mat_id_df == 0).all(axis=1), :]
        sing_nod = list(mat_id_df)

    return sing_nod


def dot(
        xnodes,
        ynodes,
        weights,
        id_mat,
        nodeff,
        color,
        base,
        colortrans,
        filename,
        base_var,
        model_dat_condition,
):
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
            if model_dat_condition and filename.startswith("IME_"):
                wei_str = "{}{}".format(utils.roundec(100 * wei), "%")  # perc
            else:
                wei_str = utils.roundec(wei)
            if id_mat[row, col] != 0 and xnode != ynode:
                col_str = color_str(wei, base, True, color, colortrans)
                dot_str += '         "{}" -> "{}" [label = "{}"{}];\n'.format(
                    xnode, ynode, wei_str, col_str
                )

    # nodes
    for i in range(xdim):
        xnode = xnodes[i]
        if xnode not in sing_nod:
            if nodeff is not None and not isnan(nodeff[i]):
                # if no nodeff given, or some elements are nan (tval ey diag)
                if model_dat_condition and filename.startswith("IME_"):
                    nodeff_str = "{}{}".format(
                        utils.roundec(100 * nodeff[i]), "%"
                    )  # perc
                else:
                    nodeff_str = utils.roundec(nodeff[i])
                col_str = color_str(nodeff[i], base, False, color, colortrans)
            else:
                nodeff_str = ""
                col_str = ""
            if base_var:  # If full_name # yyy
                xnode_show = xnode
            else:
                xnode_show = xnode
            dot_str += '         "{}"[label = "{}\\n{}"{}];\n'.format(
                xnode, xnode_show, nodeff_str, col_str
            )

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


def create_and_save_graph(
        xnodes,
        ynodes,
        x_weights_idmat_nodeff,
        y_weights_idmat_nodeff,
        color,
        dir_path,
        filename,
        base_var,
        full_name,
        model_dat_condition,
        colortrans=None,
):
    """create graph as dot string, save it as png and return it as svg"""

    form = (
        "         node [style=rounded]\n"
        "         node [shape=box]\n"
        '         ratio="compress"\n'
    )

    if color is True:
        base = compute_color_base(
            [  # absolute max over all values
                x_weights_idmat_nodeff[0],
                x_weights_idmat_nodeff[2],
                y_weights_idmat_nodeff[0],
                y_weights_idmat_nodeff[2],
            ]
        )
    elif color is None:
        base = None
    else:
        base = abs(color)  # e.g. color = 2 for t-values

    x_dot = dot(  # type: ignore
        xnodes,
        ynodes,
        *x_weights_idmat_nodeff,
        color,
        base,
        colortrans,
        filename,
        base_var,
        model_dat_condition,
    )
    y_dot = dot(  # type: ignore
        ynodes,
        ynodes,
        *y_weights_idmat_nodeff,
        color,
        base,
        colortrans,
        filename,
        base_var,
        model_dat_condition,
    )
    dot_str = "digraph { \n" + form + x_dot + y_dot + "        }"

    if full_name:
        graph_svg = utils.render_dot(dot_str)
    else:
        utils.save_graph(dir_path, filename, dot_str)
        graph_svg = utils.render_dot(dot_str, out_type="svg")

    return graph_svg


def create_graphs(graph_json, output_dir, full_name):
    """creates direct, total and mediation graph,
    for theoretical model and estimated model"""

    dir_path = output_dir
    xnodes = symbols(graph_json["xnodes"].split(","))
    ynodes = symbols(graph_json["ynodes"].split(","))
    idx = numpy_arr(graph_json["idx"])
    idy = numpy_arr(graph_json["idy"])
    show_nr_indiv = graph_json["show_nr_indiv"]
    base_var = graph_json["base_var"]
    model_dat_condition = graph_json["model_dat_condition"]

    # calculate mx_theo and my_theo
    direct_theo = graph_json["direct_theo"]

    mx_theo, my_theo = utils.directmat_alg(direct_theo, idx, idy)

    print("\nAverage and estimated graphs")

    print("ADE")
    direct_graph = create_and_save_graph(
        xnodes,
        ynodes,
        (mx_theo, idx, None),
        (my_theo, idy, None),
        False,
        dir_path,
        "ADE",
        base_var,
        full_name,
        model_dat_condition,
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
        (eyx_theo, fdx, exj_theo),
        (eyy_theo, fdy, eyj_theo),
        False,
        dir_path,
        "AME",
        base_var,
        full_name,
        model_dat_condition,
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
            (mx_hat, idx, None),
            (my_hat, idy, None),
            False,
            dir_path,
            "EDE",
            base_var,
            full_name,
            model_dat_condition,
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
            (utils.tvals(mx_hat, mx_hat_std), idx, None),
            (utils.tvals(my_hat, my_hat_std), idy, None),
            2,
            dir_path,
            "ED0",
            base_var,
            full_name,
            model_dat_condition,
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
            (eyx_hat, fdx, exj_hat),
            (eyy_hat, fdy, eyj_hat),
            False,
            dir_path,
            "EME",
            base_var,
            full_name,
            model_dat_condition,
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
            (utils.tvals(eyx_hat, eyx_hat_std), fdx, utils.tvals(exj_hat, exj_hat_std)),
            (utils.tvals(eyy_hat, eyy_hat_std), fdy, utils.tvals(eyj_hat, eyj_hat_std)),
            2,
            dir_path,
            "EM0",
            base_var,
            full_name,
            model_dat_condition,
            lambda x: abs(x),
        )

        # ED1 json
        mx_theo = numpy_arr(graph_json["mx_theo"])
        my_theo = numpy_arr(graph_json["my_theo"])
        print("ED1")
        direct_tval_graph_1 = create_and_save_graph(
            xnodes,
            ynodes,
            ((utils.tvals(mx_hat - mx_theo, mx_hat_std)), idx, None),
            ((utils.tvals(my_hat - my_theo, my_hat_std)), idy, None),
            2,
            dir_path,
            "ED1",
            base_var,
            full_name,
            model_dat_condition,
            lambda x: -abs(x),
        )

        print("EM1")
        # EM1 params
        mediation_tval_graph_1 = create_and_save_graph(
            xnodes,
            ynodes,
            (
                (utils.tvals(eyx_hat - eyx_theo, eyx_hat_std)),
                fdx,
                (utils.tvals(exj_hat - exj_theo, exj_hat_std)),
            ),
            (
                (utils.tvals(eyy_hat - eyy_theo, eyy_hat_std)),
                fdy,
                (utils.tvals(eyj_hat - eyj_theo, eyj_hat_std)),
            ),
            2,
            dir_path,
            "EM1",
            base_var,
            full_name,
            model_dat_condition,
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
            (ex_theo, edx, None),
            (ey_theo, edy, None),
            False,
            dir_path,
            "ATE",
            base_var,
            full_name,
            model_dat_condition,
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
                (ex_hat, edx, None),
                (ey_hat, edy, None),
                False,
                dir_path,
                "ETE",
                base_var,
                full_name,
                model_dat_condition,
            )
            print("ET0")
            ex_hat_std = numpy_arr(graph_json["ex_hat_std"])
            ey_hat_std = numpy_arr(graph_json["ey_hat_std"])
            total_tval_graph_0 = create_and_save_graph(
                xnodes,
                ynodes,
                (utils.tvals(ex_hat, ex_hat_std), edx, None),
                (utils.tvals(ey_hat, ey_hat_std), edy, None),
                2,
                dir_path,
                "ET0",
                base_var,
                full_name,
                model_dat_condition,
                lambda x: abs(x),
            )
            print("ET1")
            total_tval_graph_1 = create_and_save_graph(
                xnodes,
                ynodes,
                ((utils.tvals(ex_hat - ex_theo, ex_hat_std)), edx, None),
                ((utils.tvals(ey_hat - ey_theo, ey_hat_std)), edy, None),
                2,
                dir_path,
                "ET1",
                base_var,
                full_name,
                model_dat_condition,
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
            (mx_indivs[i], idx, None),
            (my_indivs[i], idy, None),
            True,
            dir_path,
            "IDE" + "_" + str(i),
            base_var,
            full_name,
            model_dat_condition,
            )
        direct_indiv_graphs.append(direct_indiv_graph)
        print("IME")
        mediation_indiv_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (eyx_indivs[i], fdx, exj_indivs[:, i]),
            (eyy_indivs[i], fdy, eyj_indivs[:, i]),
            True,
            dir_path,
            "IME" + "_" + str(i),
            base_var,
            full_name,
            model_dat_condition,
            )
        mediation_indiv_graphs.append(mediation_indiv_graph)
        print("ITE")
        total_indiv_graph = create_and_save_graph(
            xnodes,
            ynodes,
            (ex_indivs[i], edx, None),
            (ey_indivs[i], edy, None),
            True,
            dir_path,
            "ITE" + "_" + str(i),
            base_var,
            full_name,
            model_dat_condition,
            )
        total_indiv_graphs.append(total_indiv_graph)

    # render and return graph_dat

    graph_dat = {
        # average graphs
        "ADE": direct_graph,
        "ATE": total_graph,
        "AME": mediation_graph,
        # individual graphs
        "IDE": direct_indiv_graphs,
        "ITE": total_indiv_graphs,
        "IME": mediation_indiv_graphs,
    }

    if "mx_hat" in graph_json:
        # If estimate has not been done, the keys necessary
        # for these graphs are missing.

        graph_dat.update(
            {
                # estimated graphs
                "EDE": direct_hat_graph,
                "ETE": total_hat_graph,
                "EME": mediation_hat_graph,
                # tvalues graphs wrt 0
                "ED0": direct_tval_graph_0,
                "ET0": total_tval_graph_0,
                "EM0": mediation_tval_graph_0,
                # tvalues graphs wrt 1
                "ED1": direct_tval_graph_1,
                "ET1": total_tval_graph_1,
                "EM1": mediation_tval_graph_1,
            }
        )

    return graph_dat


def sym_to_str(sym_list):
    return ", ".join(str(i) for i in sym_list)


def create_json_graphs(model_dat, estimate_dat, indiv_dat):
    model_dat_condition = model_dat["final_var"] in model_dat.get("rat_var", [])

    ndim = model_dat.get("ndim", 3)
    show_total_ndim = 10  # ToDo: set globally # yyy
    is_all_graph = True if ndim < show_total_ndim else False

    model_json = {
        "dir_path": model_dat["dir_path"],
        "is_all_graph": is_all_graph,
        "company_ids": model_dat.get("company_ids", None),
        "table_company": model_dat.get("table_company", None),
        "show_nr_indiv": min(model_dat["tau"], model_dat["show_nr_indiv"]),
        "xnodes": sym_to_str(model_dat["xvars"]),
        "idx": model_dat["idx"].tolist(),
        "idy": model_dat["idy"].tolist(),
        "ynodes": sym_to_str(model_dat["yvars"]),
        "direct_theo": model_dat["direct_theo"].tolist(),
        # AME_json
        "eyx_theo": model_dat["eyx_theo"].tolist(),  # nm_array
        "eyy_theo": model_dat["eyy_theo"].tolist(),  # nm_array
        "fdx": model_dat["fdx"].tolist(),  # nm_array
        "fdy": model_dat["fdy"].tolist(),  # nm_array
        "exj_theo": model_dat["exj_theo"].tolist(),  # nm_array
        "eyj_theo": model_dat["eyj_theo"].tolist(),  # nm_array
        # ED1_json
        "mx_theo": model_dat["mx_theo"].tolist(),
        "my_theo": model_dat["my_theo"].tolist(),
        # EM1_json
        # ATE
        "ex_theo": model_dat["ex_theo"].tolist(),
        "ey_theo": model_dat["ey_theo"].tolist(),
        "edx": model_dat["edx"].tolist(),
        "edy": model_dat["edy"].tolist(),
        # indiv_dat
        "mx_indivs": [indiv.tolist() for indiv in indiv_dat["mx_indivs"]],
        "my_indivs": [indiv.tolist() for indiv in indiv_dat["my_indivs"]],
        "eyx_indivs": [indiv.tolist() for indiv in indiv_dat["eyx_indivs"]],
        "eyy_indivs": [indiv.tolist() for indiv in indiv_dat["eyy_indivs"]],
        "exj_indivs": indiv_dat["exj_indivs"].tolist(),
        "eyj_indivs": indiv_dat["eyj_indivs"].tolist(),
        "ex_indivs": [indiv.tolist() for indiv in indiv_dat["ex_indivs"]],
        "ey_indivs": [indiv.tolist() for indiv in indiv_dat["ey_indivs"]],
        "base_var": True if "base_var" in model_dat else False,
        "model_dat_condition": model_dat_condition
        # 'final_var': True if model_dat["final_var"] in model_dat["rat_var"] else False,
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
