# -*- coding: utf-8 -*-
"""causing - causal interpretation using graphs."""

from causing.utils import create_model, print_output
from causing.estimate import estimate_models
from causing.indiv import create_indiv
from causing.graph import create_graphs, create_json_graphs

def causing(model_raw_dat):
    """create graphs and reportlab model output"""

    # causing analysis
    model_dat = create_model(model_raw_dat)
    estimate_dat = estimate_models(model_dat)
    indiv_dat = create_indiv(model_dat)
    
    print_output(model_dat, estimate_dat, indiv_dat)


    graph_json = create_json_graphs(model_dat, estimate_dat, indiv_dat)


    # show total graphs only for smaller ndim
    ndim = model_dat.get('ndim', None)
    show_total_ndim = 10  # ToDo: set globally # yyy
    if ndim < show_total_ndim:
        ndim < show_total_ndim
        graph_dat = create_graphs(graph_json)
    else:
        graph_dat = None



    analyze_dat = {
        "model_dat": model_dat,
        "estimate_dat": estimate_dat,
        "indiv_dat": indiv_dat,
        "graph_dat": graph_dat,
        }
    
    return analyze_dat

# if __name__ == "__main__":
#
#     model_raw_dat = models.example()
#
#     analyze_dat = causing(model_raw_dat)

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
