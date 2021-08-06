from dataclasses import dataclass, field
from typing import List

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

        self.m_pair = utils.make_partial_diffs(
            self.xvars, self.yvars, self.equations, self.model_lam
        )

    def compute(self, xdat: np.array) -> np.array:
        xdat = np.array(xdat).reshape(len(self.xvars), -1)
        yhat = np.array([self.model_lam(*xval) for xval in xdat.T]).T
        return yhat
