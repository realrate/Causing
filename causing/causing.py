# -*- coding: utf-8 -*-
"""causing - causal interpretation using graphs."""

from causing import models
from causing import estimate
from causing import indiv
from causing import graph
from causing import report
from causing import utils


def causing(model_raw_dat):
    """create graphs and reportlab model output"""

    # causing analysis
    model_dat = utils.create_model(model_raw_dat)
    estimate_dat = estimate.estimate_models(model_dat)
    indiv_dat = indiv.create_indiv(model_dat)
    utils.print_output(model_dat, estimate_dat, indiv_dat)
    graph_dat = graph.create_graphs(model_dat, estimate_dat, indiv_dat)

    analyze_dat = {
        "model_dat": model_dat,
        "estimate_dat": estimate_dat,
        "indiv_dat": indiv_dat,
        "graph_dat": graph_dat,
        }

    # create pdf output files
    report.average_and_estimated_effects(analyze_dat)
    report.tvalues_and_biases(analyze_dat)
    for individual_id in range(min(model_dat["tau"],
                                   model_dat["show_nr_indiv"])):
        report.mediation_effects(analyze_dat, individual_id)
    
    return analyze_dat

# if __name__ == "__main__":

#     model_dat = models.example()

#     analyze_dat = causing(model_dat)

# =============================================================================
#     # profiling
#     import cProfile
#     import pstats
#     cProfile.runctx("causing(model_dat)", globals(), locals(), "output/profile.txt")
#     # print analysis of existing profile.txt
#     prof = pstats.Stats("output/profile.txt")                   # load
#     prof.sort_stats('cumulative').print_stats(20)               # sorted cumulative
#     prof.strip_dirs().sort_stats('time').print_stats(".py", 20) # sorted internal
#     prof.print_callers(".py", 20)                               # functions calling
# =============================================================================
