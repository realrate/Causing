import unittest

import numpy as np
from sympy import symbols

import causing.bias
from causing.model import Model


class TestBias(unittest.TestCase):
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
    ymvars = [Y3]

    def test_no_bias(self):
        ymdat = np.array([[4, 4, 4, 3.9, 4.01]])
        biases, biases_std = causing.bias.estimate_biases(
            self.m, self.xdat, self.ymvars, ymdat
        )
        self.assertAlmostEqual(biases[0], 0, places=1)
        self.assertAlmostEqual(biases[1], 0, places=1)
        self.assertAlmostEqual(biases[2], 0, places=1)

    def test_bias(self):
        ymdat = np.array([[5, 5, 5, 4.9, 5.01]])
        biases, biases_std = causing.bias.estimate_biases(
            self.m, self.xdat, self.ymvars, ymdat
        )
        self.assertAlmostEqual(biases[0], 0.32, places=2)
        self.assertAlmostEqual(biases[1], 0.966, places=3)
        self.assertAlmostEqual(biases[2], 0.966, places=3)


class TestBiasInvariant(unittest.TestCase):
    xdat = np.array(
        [
            [1, 1, 1.01, 1.02, 0.99],
            [1, 1.01, 1, 1.03, 0.98],
        ]
    )

    def test_bias_invariant(self):
        X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])
        for bias in (0, 10, 100):
            with self.subTest(bias=bias):
                equations = (
                    X1,
                    bias + X2 + 2 * Y1,
                    Y1 + Y2,
                )
                m = Model(
                    xvars=[X1, X2],
                    yvars=[Y1, Y2, Y3],
                    equations=equations,
                    final_var=Y3,
                )
                ymvars = [Y3]
                ymdat = np.array([[4, 4, 4, 3.9, 4.01]])
                biases, biases_std = causing.bias.estimate_biases(
                    m, self.xdat, ymvars, ymdat
                )
                self.assertAlmostEqual(biases[1], -bias, places=1)

    def test_bias_invariant_quotient(self):
        """This simple estimation fails with the SLSQP method"""
        X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])
        for bias in (0, 5, 20):
            with self.subTest(bias=bias):
                equations = (
                    bias + 3,
                    1 / Y1,
                )
                m = Model(
                    xvars=[X1, X2],
                    yvars=[Y1, Y2],
                    equations=equations,
                    final_var=Y2,
                )
                ymvars = [Y2]
                ymdat = np.array([[1 / 3, 1 / 3, 1 / 3, 1 / 2.9, 1 / 3.01]])
                biases, biases_std = causing.bias.estimate_biases(
                    m, self.xdat, ymvars, ymdat
                )
                self.assertAlmostEqual(biases[0], -bias, places=1)
