from dataclasses import dataclass, field
from typing import List, Tuple

import sympy
import numpy as np

from causing import utils


@dataclass
class Model:

    xvars: List[sympy.Symbol]
    yvars: List[sympy.Symbol]
    equations: List[sympy.Expr]

    ndim: int = field(init=False)
    mdim: int = field(init=False)

    def __post_init__(self):
        self.ndim = len(self.yvars)
        self.mdim = len(self.xvars)

        substituted_eqs = list(self.equations)
        for i, yvar in enumerate(self.yvars):
            for j in range(i + 1, len(self.yvars)):
                substituted_eqs[j] = substituted_eqs[j].subs(yvar, substituted_eqs[i])
        self.model_lam = sympy.lambdify(
            self.xvars, substituted_eqs, modules=["sympy", "numpy"]
        )

        mx_alg, my_alg, *self.m_pair = make_partial_diffs(
            self.xvars, self.yvars, self.equations, self.model_lam
        )

        # identification matrices for direct effects
        self.idx = utils.digital(mx_alg)
        self.idy = utils.digital(my_alg)

        # effect identification matrices
        self.edx, self.edy = utils.compute_ed(self.idx, self.idy)

    def compute(self, xdat: np.array) -> np.array:
        xdat = np.array(xdat).reshape(len(self.xvars), -1)
        yhat = np.array([self.model_lam(*xval) for xval in xdat.T]).T
        return yhat


def make_partial_diffs(
    xvars, yvars, equations, model_lam
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Create partial derivatives for model in adj matrix form"""
    mx_alg = np.array([[sympy.diff(eq, xvar) for xvar in xvars] for eq in equations])
    my_alg = np.array([[sympy.diff(eq, yvar) for yvar in yvars] for eq in equations])

    mx_alg[mx_alg == 0] = float("NaN")
    my_alg[my_alg == 0] = float("NaN")

    modules = ["sympy", "numpy"]
    mx_lamxy = sympy.lambdify((xvars, yvars), mx_alg, modules=modules)
    my_lamxy = sympy.lambdify((xvars, yvars), my_alg, modules=modules)

    def mx_lam(xval):
        yval = model_lam(*list(xval.T))
        return mx_lamxy(xval, yval)

    def my_lam(xval):
        yval = model_lam(*list(xval.T))
        return my_lamxy(xval, yval)

    return (mx_alg, my_alg, mx_lam, my_lam)
