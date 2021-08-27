from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from functools import cached_property

import sympy
import numpy as np

from causing import utils


@dataclass
class Model:

    xvars: List[sympy.Symbol]
    yvars: List[sympy.Symbol]
    ymvars: List[sympy.Symbol]
    equations: Tuple[sympy.Expr, ...]
    final_var: sympy.Symbol

    ndim: int = field(init=False)
    mdim: int = field(init=False)

    def __post_init__(self):
        self.mdim = len(self.xvars)
        self.ndim = len(self.yvars)

        self.biases = [sympy.Symbol(f"bias{i+1}") for i in range(len(self.yvars))]

        mx_alg, my_alg, *self.m_pair = self._make_partial_diffs()

        # identification matrices for direct effects
        self.idx = utils.digital(mx_alg)
        self.idy = utils.digital(my_alg)

        # effect identification matrices
        self.edx, self.edy = utils.compute_ed(self.idx, self.idy)

        # final identification matrices
        self.fdx, self.fdy = utils.compute_fd(
            self.idx, self.idy, self.yvars, self.final_var
        )

        # more dimensions
        self.pdim = len(self.ymvars)
        self.qxdim = utils.count_nonzero(self.idx)
        self.qydim = utils.count_nonzero(self.idy)
        self.qdim = self.qxdim + self.qydim

    def compute(self, xdat: np.array, bias: float = 0, bias_ind: int = 0) -> np.array:
        """Compute y values for given x values

        xdat: m rows, tau columns
        returns: n rows, tau columns
        """
        assert isinstance(bias, (float, int)), f"bias must be scalar, not {bias!r}"
        assert xdat.ndim == 2, f"xdat must be m*tau (is {xdat.ndim}-dimensional)"
        assert xdat.shape[0] == self.mdim, f"xdat must be m*tau (is {xdat.shape})"
        bias_dat = [bias if i == bias_ind else 0 for i in range(len(self.biases))]
        yhat = np.array([self._model_lam(*xval, *bias_dat) for xval in xdat.T]).T
        assert yhat.shape == (self.ndim, xdat.shape[1])
        return yhat

    def theo(self, xval: np.array) -> Dict[str, np.array]:
        mx_lam, my_lam = self.m_pair

        # numeric direct effects since no sympy algebraic derivative
        mx_theo = utils.replace_heaviside(
            np.array(mx_lam(xval)), self.xvars, xval
        )  # yyy
        my_theo = utils.replace_heaviside(
            np.array(my_lam(xval)), self.xvars, xval
        )  # yyy

        # total and final effects
        ex_theo, ey_theo = utils.total_effects_alg(mx_theo, my_theo, self.edx, self.edy)
        exj_theo, eyj_theo, eyx_theo, eyy_theo = utils.compute_mediation_effects(
            mx_theo,
            my_theo,
            ex_theo,
            ey_theo,
            self.yvars,
            self.final_var,
        )

        return dict(
            mx_theo=mx_theo,
            my_theo=my_theo,
            ex_theo=ex_theo,
            ey_theo=ey_theo,
            exj_theo=exj_theo,
            eyj_theo=eyj_theo,
            eyx_theo=eyx_theo,
            eyy_theo=eyy_theo,
        )

    def _make_partial_diffs(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Create partial derivatives for model in adj matrix form"""
        mx_alg = np.array(
            [[sympy.diff(eq, xvar) for xvar in self.xvars] for eq in self.equations]
        )
        my_alg = np.array(
            [[sympy.diff(eq, yvar) for yvar in self.yvars] for eq in self.equations]
        )

        mx_alg[mx_alg == 0] = float("NaN")
        my_alg[my_alg == 0] = float("NaN")

        modules = ["sympy", "numpy"]
        mx_lamxy = sympy.lambdify((self.xvars, self.yvars), mx_alg, modules=modules)
        my_lamxy = sympy.lambdify((self.xvars, self.yvars), my_alg, modules=modules)

        def mx_lam(xval):
            xdat = np.vstack(xval)
            yval = self.compute(xdat)[:, 0]
            return mx_lamxy(xval, yval)

        def my_lam(xval):
            xdat = np.vstack(xval)
            yval = self.compute(xdat)[:, 0]
            return my_lamxy(xval, yval)

        return (mx_alg, my_alg, mx_lam, my_lam)

    @cached_property
    def _model_lam(self):
        eqs_with_bias = [eq + bias for eq, bias in zip(self.equations, self.biases)]
        substituted_eqs = list(eqs_with_bias)
        for i, yvar in enumerate(self.yvars):
            for j in range(i + 1, len(self.yvars)):
                substituted_eqs[j] = substituted_eqs[j].subs(yvar, substituted_eqs[i])
        return sympy.lambdify(
            self.xvars + self.biases, substituted_eqs, modules=["sympy", "numpy"]
        )
