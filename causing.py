# -*- coding: utf-8 -*-
"""causing - causal interpretation using graphs."""

import sys
import models
import estimate
import indiv
import graph
import report
import utils


def causing(model_dat):
    """create graphs and reportlab model output"""

    # print to file
    stdout = sys.stdout
    fha = open(model_dat["dir_path"] + "output.txt", 'w')
    sys.stdout = fha

    # causing analysis
    model_dat = utils.create_model(model_dat)
    estimate_dat = estimate.estimate_models(model_dat)
    indiv_dat = indiv.create_indiv(model_dat)
    utils.print_output(model_dat, estimate_dat, indiv_dat)
    graph_dat = graph.create_graphs(model_dat, estimate_dat, indiv_dat)

    analyze_dat = {
        "model_dat": model_dat,
        "model_dat": model_dat,
        "estimate_dat": estimate_dat,
        "indiv_dat": indiv_dat,
        "graph_dat": graph_dat,
        }

    # create pdf output files
    report.average_and_estimated_effects(analyze_dat)
    report.tvalues_and_biases(analyze_dat)
    for individual_id in range(min(model_dat["tau"], model_dat["show_nr_indiv"])):
        report.mediation_effects(analyze_dat, individual_id)

    sys.stdout = stdout
    fha.close()

if __name__ == "__main__":

    model_dat = models.example()

    causing(model_dat)

# =============================================================================
#     # start analyze with profiling
#     import cProfile
#     import pstats
#     cProfile.runctx("causing()", globals(), locals(), "output/profile.txt")
#     prof = pstats.Stats("output/profile.txt")
#     prof.strip_dirs()               # strip directory paths
#     prof.sort_stats('time')         # sort by internal time
#     prof.print_stats(".py", 20)     # print 20 .py functions
#     prof.print_callers(".py", 20)   # functions that called the above functions
# =============================================================================
