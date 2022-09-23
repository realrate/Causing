# -*- coding: utf-8 -*-
"""causing - causal interpretation using graphs."""

# flake8: noqa

# public Causing API
from causing.model import Model


def create_indiv(m: Model, xdat, show_nr_indiv: int) -> dict:
    """create indiv analysis data for mediation indiv graph values,
    using individual total effects and mediation effects"""
    eff = m.calc_effects(xdat)
    for key in ["exj_indivs", "eyj_indivs", "eyx_indivs", "eyy_indivs"]:
        if key in ["exj_indivs", "eyj_indivs"]:
            eff[key] = eff[key][:, :show_nr_indiv]
        else:
            eff[key] = eff[key][:show_nr_indiv]
    return eff
