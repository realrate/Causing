import unittest

import numpy as np
from sympy import symbols

from causing.model import Model
from causing import estimate


class TestBias(unittest.TestCase):
    def test_bias(self):
        X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])
        equations = (
            X1,
            X2 + 2 * Y1,
            Y1 + Y2,
        )
        m = Model(
            xvars=[X1, X2],
            yvars=[Y1, Y2, Y3],
            equations=equations,
            final_var=Y3,
        )

        xdat = np.array(
            [
                [1, 1, 1.01, 1.02, 0.99],
                [1, 1.01, 1, 1.03, 0.98],
            ]
        )
        estimate_input = dict(
            ymvars=[Y3],
            ymdat=np.array([[5, 5, 5, 4.9, 5.01]]),
            estimate_bias=True,
            alpha=None,
            dof=None,
        )

        mean_theo = m.theo(xdat.mean(axis=1))
        estimate_dat = estimate.estimate_models(m, xdat, mean_theo, estimate_input)

        biases = estimate_dat["biases"]
        self.assertAlmostEqual(biases[0], 0.32, places=2)
        self.assertAlmostEqual(biases[1], 0.966, places=3)
        self.assertAlmostEqual(biases[2], 0.966, places=3)
