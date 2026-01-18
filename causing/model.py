from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Callable
from functools import cached_property
import networkx

import sympy
import numpy as np


class NumericModelError(Exception):
    pass


@dataclass
class Model:

    xvars: list[str]
    yvars: list[str]
    equations: Sequence[sympy.Expr]
    final_var: str
    parameters: dict[str, float] = field(default_factory=dict)

    ndim: int = field(init=False)
    mdim: int = field(init=False)
    graph: networkx.DiGraph = field(init=False)

    def __post_init__(self):
        # If sympy.Symbols are passed in, convert to string
        self.xvars = [str(var) for var in self.xvars]
        self.yvars = [str(var) for var in self.yvars]
        self.final_var = str(self.final_var)

        self.mdim = len(self.xvars)
        self.ndim = len(self.yvars)

        self.graph = networkx.DiGraph()
        for yvar, eq in zip(self.yvars, self.equations):
            if isinstance(eq, (float, int)):
                continue
            for sym in eq.free_symbols:
                if str(sym) in self.parameters:
                    continue
                self.graph.add_edge(str(sym), yvar)
        for var in self.vars:
            self.graph.add_node(var)
        self.trans_graph = networkx.transitive_closure(self.graph, reflexive=True)

    @np.errstate(all="raise")
    def compute(
        self,
        xdat: np.array,
        fixed_yval: np.array = None,
        fixed_yind: int = None,
        fixed_from_ind: int = None,
        fixed_to_yind: int = None,
        fixed_vals: list = None,
        parameters: dict[str, float] = {},
    ) -> np.array:
        """Compute y values for given x values (Optimized Version)
        xdat: m rows, tau columns
        returns: n rows, tau columns
        """

        assert xdat.ndim == 2, f"xdat must be m*tau (is {xdat.ndim}-dimensional)"
        assert xdat.shape[0] == self.mdim, f"xdat must be m*tau (is {xdat.shape})"
        tau = xdat.shape[1]
        parameters = self.parameters | parameters

        # Use np.full for clarity and potential small performance gain
        yhat = np.full((self.ndim, tau), np.nan)

        for i, eq in enumerate(self._model_lam):

            if fixed_yind == i:
                yhat[i, :] = fixed_yval
            else:
                # 1. Build inputs (Vectorized)
                # Use np.vstack for efficient vertical stacking.
                # `yhat` will have NaNs for unsolved variables, which is correct.
                eq_inputs = np.vstack([xdat, yhat])

                # 2. Apply fixed values if needed (Vectorized)
                if fixed_to_yind == i:
                    # Directly modify the correct "row" in the input matrix.
                    # This is much cleaner and faster.
                    eq_inputs[fixed_from_ind, :] = fixed_vals

                # 3. Evaluate equation (Vectorized)
                try:
                    np.seterr(under="ignore")

                    # This is the core optimization. We unpack the rows of `eq_inputs`
                    # as separate arguments into the lambdified function. NumPy will
                    # then compute the results for all `tau` columns at once.
                    yhat[i, :] = eq(*eq_inputs, *parameters.values())

                except Exception as e:
                    raise NumericModelError(
                        f"Failed to compute model value for yvar {self.yvars[i]}: {e}"
                    ) from e

        return yhat

    def calc_effects(self, xdat: np.array, xdat_mean=None, yhat_mean=None):
        """Calculate node and edge effects for the given input

        Pass mean values only if you compute effects for a subset of the
        individuals you want to use as a benchmark.
        """
        yhat = self.compute(xdat)
        if yhat_mean is None:
            yhat_mean = np.mean(yhat, axis=1)
        if xdat_mean is None:
            xdat_mean = np.mean(xdat, axis=1)
        tau = xdat.shape[1]
        exj = np.full([len(self.xvars), tau], float("NaN"))
        eyx = np.full([tau, len(self.yvars), len(self.xvars)], float("NaN"))
        for xind, xvar in enumerate(self.xvars):
            if not self.trans_graph.has_edge(xvar, self.final_var):
                # Without path to final_var, there is no effect on final_var
                continue

            fixed_xdat = xdat.copy()
            fixed_xdat[xind, :] = xdat_mean[xind]
            fixed_yhat = self.compute(fixed_xdat)
            exj[xind, :] = yhat[self.final_ind] - fixed_yhat[self.final_ind]

            for yind, yvar in enumerate(self.yvars):
                if not self.graph.has_edge(xvar, yvar):
                    # Without edge, there is no mediated effect for that edge
                    continue
                if not self.trans_graph.has_edge(yvar, self.final_var):
                    # Without path to final_var, there is no effect on final_var
                    continue

                fixed_vals = fixed_xdat.T[:, xind]
                eyx[:, yind, xind] = (
                    yhat[self.final_ind]
                    - self.compute(
                        xdat,
                        fixed_from_ind=xind,
                        fixed_to_yind=yind,
                        fixed_vals=fixed_vals,
                    )[self.final_ind]
                )

        eyj = np.full([len(self.yvars), tau], float("NaN"))
        eyy = np.full([tau, len(self.yvars), len(self.yvars)], float("NaN"))
        for yind, yvar in enumerate(self.yvars):
            if not self.trans_graph.has_edge(yvar, self.final_var):
                # Without path to final_var, there is no effect on final_var
                continue

            fixed_yval = yhat_mean[yind]
            fixed_yhat = self.compute(xdat, fixed_yind=yind, fixed_yval=fixed_yval)
            eyj[yind, :] = yhat[self.final_ind] - fixed_yhat[self.final_ind]

            for yind2, yvar2 in enumerate(self.yvars):
                if not self.graph.has_edge(yvar, yvar2):
                    # Without edge, there is no mediated effect for that edge
                    continue
                if not self.trans_graph.has_edge(yvar2, self.final_var):
                    # Without path to final_var, there is no effect on final_var
                    continue

                fixed_vals = fixed_yhat.T[:, yind]
                eyy[:, yind2, yind] = (
                    yhat[self.final_ind]
                    - self.compute(
                        xdat,
                        fixed_from_ind=len(self.xvars) + yind,
                        fixed_to_yind=yind2,
                        fixed_vals=fixed_vals,
                    )[self.final_ind]
                )

        return {
            # model results
            "yhat": yhat,
            # nodes
            "xnodeeffects": exj,
            "ynodeeffects": eyj,
            # edges
            "xedgeeffects": eyx,
            "yedgeeffects": eyy,
        }

    @cached_property
    def _model_lam(self) -> Sequence[Callable]:
        """Create lambdified equations with NumPy-compatible functions."""
        lambdas = []
        ordered_vars = self.vars + list(self.parameters.keys())

        # Define placeholder for vectorized max function
        vectorized_max = sympy.Function("vectorized_max")

        # Define custom translation mapping
        custom_modules = [{"vectorized_max": np.maximum}, "numpy"]

        for i, eq in enumerate(self.equations):
            # Replace sympy.Max with our placeholder
            fixed_eq = eq.subs(sympy.Max, vectorized_max)

            # Lambdify with custom NumPy mapping
            lam = sympy.lambdify(ordered_vars, fixed_eq, modules=custom_modules)
            lambdas.append(lam)

        return lambdas

    @cached_property
    def final_ind(self):
        "Index of final variable"
        return self.yvars.index(self.final_var)

    @property
    def vars(self) -> list[str]:
        return self.xvars + self.yvars

    def shrink(m: Model, remove_nodes) -> Model:  # noqa
        """Create a model  without `remove_nodes`"""
        yvars = []
        equations = []
        substitutions: list[tuple] = []
        for yvar, eq in zip(m.yvars, m.equations):
            if yvar in remove_nodes:
                substitutions.insert(0, (yvar, eq))
            else:
                yvars.append(yvar)
                equations.append(eq.subs(substitutions))

        new_model = Model(m.xvars, yvars, equations, m.final_var, m.parameters)
        return new_model
