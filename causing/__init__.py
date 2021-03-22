# -*- coding: utf-8 -*-
"""causing - causal interpretation using graphs."""

from causing.utils import create_model, print_output
from causing.estimate import estimate_models
from causing.indiv import create_indiv
from causing.graph import create_json_graphs


def analyze(model_raw_dat):
    model_dat = create_model(model_raw_dat)
    estimate_dat = estimate_models(model_dat)
    indiv_dat = create_indiv(model_dat)

    graphs = create_json_graphs(model_dat, estimate_dat, indiv_dat)
    return {
        "model_dat": model_dat,
        "estimate_dat": estimate_dat,
        "indiv_dat": indiv_dat,
        "graph_dat": graphs,
    }
