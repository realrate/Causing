# -*- coding: utf-8 -*-
"""Create direct, total and mediation Graphviz graph from dot_str"""
import re
import locale
import subprocess
from typing import Iterable
from itertools import chain
from pathlib import Path
from functools import cache

import numpy as np
import networkx

from causing.model import Model


DOT_COMMAND = "dot"


@cache
def dot_version() -> list[int]:
    full_output = subprocess.check_output(
        [DOT_COMMAND, "-V"], encoding="utf-8", stderr=subprocess.STDOUT
    )
    match = re.search(r"(\d+[.])+\d+", full_output)
    assert match
    version = match.group(0)
    return [int(x) for x in version.split(".")]


def fix_svg_scale(svg_code):
    """Work around graphviz SVG generation bug

    Graphviz divides the numbers in the viewBox attribute by the scale instead
    of multiplying it. We work around it my multiplying with the scale twice.
    See https://github.com/realrate/RealRate-Private/issues/631
    and https://gitlab.com/graphviz/graphviz/-/issues/1406
    """
    # find scaling factor
    scale_match = re.search(r"scale\(([0-9.]+)", svg_code)
    assert scale_match
    factor = float(scale_match.group(1)) ** 2

    # edit SVG tag
    orig_svg_tag = next(re.finditer(r"<svg.*?>", svg_code, flags=re.DOTALL)).group(0)
    attrs = {
        match.group(1): match.group(2)
        for match in re.finditer(r'(\w+)="(.*?)"', orig_svg_tag)
    }
    new_svg_tag = "<svg "
    for attr, val in attrs.items():
        if attr == "viewBox":
            val = " ".join(str(float(x) * factor) for x in val.split(" "))
        new_svg_tag += f'{attr}="{val}" '
    new_svg_tag += ">"

    # return with changed SVG tag
    return svg_code.replace(orig_svg_tag, new_svg_tag)


def save_graph(path: Path, graph_dot):
    """Render graph as SVG"""

    path.parent.mkdir(parents=True, exist_ok=True)

    # with open(path.stem + ".txt", "w") as file:
    #     file.write(graph_dot)

    svg_code = subprocess.check_output(
        [DOT_COMMAND, "-Tsvg"], input=graph_dot, encoding="utf-8"
    )
    if dot_version()[0] < 3:
        svg_code = fix_svg_scale(svg_code)
    with open(path, "w") as f:
        f.write(svg_code)


def annotated_graphs(
    m: Model,
    graph_json,
    ids: Iterable[str],
    node_labels: dict[str, str] = {},
) -> Iterable[networkx.DiGraph]:
    """Return DiGraphs with all information required to draw IME graphs"""
    for graph_id, exj, eyj, eyx, eyy in zip(
        ids,
        np.array(graph_json["exj_indivs"]).T,
        np.array(graph_json["eyj_indivs"]).T,
        graph_json["eyx_indivs"],
        graph_json["eyy_indivs"],
    ):
        g = m.graph.copy()
        g.graph["id"] = graph_id

        # nodes
        for var, effect in chain(zip(m.xvars, exj), zip(m.yvars, eyj)):
            if np.isnan(effect):
                g.remove_node(var)
                continue
            data = g.nodes[var]
            data["effect"] = effect
            data["label"] = node_labels.get(var, var)

        # edges
        for to_node, x_effects, y_effects in zip(m.yvars, eyx, eyy):
            for from_node, eff in chain(
                zip(m.xvars, x_effects), zip(m.yvars, y_effects)
            ):
                if not np.isnan(eff):
                    g[from_node][to_node]["effect"] = eff

        yield g


NODE_PALETTE = "#ff7973 #FFC7AD #EEEEEE #BDE7BD #75cf73".split(" ")
EDGE_PALETTE = "#ff7973 #FFC7AD #BBBBBB #aad3aa #75cf73".split(" ")
PEN_WIDTH_PALETTE = [6, 4, 2, 4, 6]


def color(val, max_val, palette):
    """Choose element of palette based on `val`

    val == -max_val will return palette[0]
    val == +max_val will return palette[-1]
    """
    zero_one_scale = (val + max_val) / (2 * max_val)
    ind = round(zero_one_scale * len(palette) - 0.5)
    clipped_ind = np.clip(round(ind), 0, len(palette) - 1)
    return palette[clipped_ind]


def graph_to_dot(g: networkx.DiGraph, invisible_edges):
    dot_str = """digraph {
    node [style="filled,rounded"]
    node [shape=box]
    node [color="#444444"]
    ratio="compress"
    size="8,10"
"""
    max_val = max(
        [abs(data["effect"]) for _, data in g.nodes(data=True)]
        + [abs(data["effect"]) for _, _, data in g.edges(data=True)]
    )

    for node, data in g.nodes(data=True):
        eff_str = locale.format_string("%.2f%%", data["effect"] * 100)
        label = data["label"].replace("\n", r"\n") + r"\n" + eff_str
        col_str = color(data["effect"], max_val, palette=NODE_PALETTE)
        dot_str += f'    "{node}"[label = "{label}" fillcolor="{col_str}"]\n'

    for from_node, to_node, data in g.edges(data=True):
        eff_str = locale.format_string("%.2f%%", data["effect"] * 100)
        col_str = color(data["effect"], max_val, palette=EDGE_PALETTE)
        penwidth = color(data["effect"], max_val, palette=PEN_WIDTH_PALETTE)
        dot_str += f'    "{from_node}" -> "{to_node}" [label="{eff_str}" color="{col_str}" penwidth="{penwidth}"]\n'

    for from_node, to_node in invisible_edges:
        dot_str += f'    "{from_node}" -> "{to_node}" [style = "invisible", arrowhead="none"]\n'

    dot_str += "}"
    return dot_str


def create_graphs(
    graphs: Iterable[networkx.DiGraph], output_dir: Path, invisible_edges=()
):
    for g in graphs:
        filename = f"IME_{g.graph['id']}.svg"
        print("Create", filename)
        dot_str = graph_to_dot(g, invisible_edges)
        save_graph(output_dir / filename, dot_str)
