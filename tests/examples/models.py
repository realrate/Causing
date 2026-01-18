import unittest
import numpy as np
from sympy import symbols, Matrix

from causing.examples.models import example, education


def compute_theo_effects(m, xpoint):
    """
    Compute theoretical effects at a given point using analytical derivatives.
    This recreates the functionality of the old theo() method.

    Args:
        m: Model object
        xpoint: 1-D array of x values at which to evaluate, with length m.mdim

    Returns:
        Dictionary with effect matrices (mx_theo, my_theo, ex_theo, ey_theo,
        exj_theo, eyj_theo, eyx_theo, eyy_theo)

    Notes:
        - Uses symbolic differentiation via sympy to compute Jacobian matrices
        - Solves for total effects using matrix inversion: (I - dY/dY)^(-1) * dY/dX
        - Falls back to a least-squares approximate solution if the system is singular or rank-deficient
    """
    # Create symbolic variables
    xvars_sym = symbols(m.xvars)
    yvars_sym = symbols(m.yvars)

    # Compute ypoint
    ypoint = m.compute(xpoint.reshape(-1, 1)).flatten()
    point_dict = {str(xvars_sym[i]): xpoint[i] for i in range(len(xvars_sym))}
    point_dict.update({str(yvars_sym[i]): ypoint[i] for i in range(len(yvars_sym))})

    # Create vectors for differentiation
    xvec = Matrix(xvars_sym)
    yvec = Matrix(yvars_sym)
    eq_vec = Matrix(list(m.equations))

    # Compute Jacobian matrices
    # mx_theo: dY/dX direct (partial derivatives)
    mx_jacob = eq_vec.jacobian(xvec)
    mx_theo = np.array(mx_jacob.subs(point_dict)).astype(np.float64)

    # my_theo: dY/dY direct (partial derivatives)
    my_jacob = eq_vec.jacobian(yvec)
    my_theo = np.array(my_jacob.subs(point_dict)).astype(np.float64)

    # For total effects, solve: (I - dY/dY) * dY/dX_total = dY/dX_direct
    matrix_I = np.eye(m.ndim)
    try:
        ex_theo = np.linalg.solve(matrix_I - my_theo, mx_theo)
    except np.linalg.LinAlgError:
        ex_theo = np.linalg.lstsq(matrix_I - my_theo, mx_theo, rcond=None)[0]

    # ey_theo: total effects of Y on Y
    try:
        ey_theo = np.linalg.solve(matrix_I - my_theo, matrix_I)
    except np.linalg.LinAlgError:
        ey_theo = np.linalg.lstsq(matrix_I - my_theo, matrix_I, rcond=None)[0]

    # Final effects (on the final variable)
    final_ind = m.yvars.index(m.final_var)
    xnodeeffect_theo = ex_theo[final_ind, :]
    ynodeeffect_theo = ey_theo[final_ind, :]

    # Mediation effects
    # xedgeeffect: mediation through Y for each X->Y edge
    # xedgeeffect[y, x] represents the effect of X on the final variable, mediated through Y
    # Formula: xedgeeffect[y, x] = mx[y, x] * ynodeeffect[y]
    xedgeeffect_theo = np.full((m.ndim, m.mdim), np.nan)
    for yind in range(m.ndim):
        for xind in range(m.mdim):
            if mx_theo[yind, xind] != 0 and not np.isnan(mx_theo[yind, xind]):
                xedgeeffect_theo[yind, xind] = (
                    mx_theo[yind, xind] * ynodeeffect_theo[yind]
                )

    # yedgeeffect: mediation through Y->Y edges
    # yedgeeffect[y2, y1] represents the effect of Y1 on the final variable, mediated through the Y1->Y2 edge
    # Formula: yedgeeffect[y2, y1] = my[y2, y1] * ynodeeffect[y2]
    yedgeeffect_theo = np.full((m.ndim, m.ndim), np.nan)
    for yind1 in range(m.ndim):
        for yind2 in range(m.ndim):
            if my_theo[yind2, yind1] != 0 and not np.isnan(my_theo[yind2, yind1]):
                yedgeeffect_theo[yind2, yind1] = (
                    my_theo[yind2, yind1] * ynodeeffect_theo[yind2]
                )

    # Replace 0 with NaN where there's no edge in the graph
    for yind in range(m.ndim):
        for xind in range(m.mdim):
            if not m.graph.has_edge(m.xvars[xind], m.yvars[yind]):
                mx_theo[yind, xind] = np.nan
                xedgeeffect_theo[yind, xind] = np.nan
            # Also set ex_theo to NaN where there's no transitive path
            if not m.trans_graph.has_edge(m.xvars[xind], m.yvars[yind]):
                ex_theo[yind, xind] = np.nan

    for yind1 in range(m.ndim):
        for yind2 in range(m.ndim):
            if not m.graph.has_edge(m.yvars[yind1], m.yvars[yind2]):
                my_theo[yind2, yind1] = np.nan
                yedgeeffect_theo[yind2, yind1] = np.nan
            # Also set ey_theo to NaN where there's no transitive path
            if not m.trans_graph.has_edge(m.yvars[yind1], m.yvars[yind2]):
                ey_theo[yind2, yind1] = np.nan

    # Set to NaN where there's no path to final var
    for xind in range(m.mdim):
        if not m.trans_graph.has_edge(m.xvars[xind], m.final_var):
            xnodeeffect_theo[xind] = np.nan

    for yind in range(m.ndim):
        if not m.trans_graph.has_edge(m.yvars[yind], m.final_var):
            ynodeeffect_theo[yind] = np.nan

    return {
        "mx_theo": mx_theo,
        "my_theo": my_theo,
        "ex_theo": ex_theo,
        "ey_theo": ey_theo,
        "xnodeeffect_theo": xnodeeffect_theo,
        "ynodeeffect_theo": ynodeeffect_theo,
        "xedgeeffect_theo": xedgeeffect_theo,
        "yedgeeffect_theo": yedgeeffect_theo,
    }


class TestExampleModels(unittest.TestCase):
    def test_example(self):
        """Checks coefficient matrices for direct, total and final effects of example."""
        m, xdat = example()
        generated_theo = compute_theo_effects(m, xdat.mean(axis=1))

        # direct effects
        mx_theo = np.array([[1, "NaN"], ["NaN", 1], ["NaN", "NaN"]]).astype(np.float64)
        my_theo = np.array(
            [["NaN", "NaN", "NaN"], [11.92914837, "NaN", "NaN"], [1, 1, "NaN"]]
        ).astype(np.float64)

        # total effects
        ex_theo = np.array([[1, "NaN"], [11.92914837, 1], [12.92914837, 1]]).astype(
            np.float64
        )
        ey_theo = np.array(
            [[1, "NaN", "NaN"], [11.92914837, 1, "NaN"], [12.92914837, 1, 1]]
        ).astype(np.float64)

        # final effects
        xnodeeffect_theo = np.array([12.92914837, 1]).astype(np.float64)
        ynodeeffect_theo = np.array([12.92914837, 1, 1]).astype(np.float64)
        xedgeeffect_theo = np.array(
            [[12.92914837, "NaN"], ["NaN", 1], ["NaN", "NaN"]]
        ).astype(np.float64)
        yedgeeffect_theo = np.array(
            [["NaN", "NaN", "NaN"], [11.92914837, "NaN", "NaN"], [1, 1, "NaN"]]
        ).astype(np.float64)

        expected_theo = dict(
            mx_theo=mx_theo,
            my_theo=my_theo,
            ex_theo=ex_theo,
            ey_theo=ey_theo,
            xnodeeffect_theo=xnodeeffect_theo,
            ynodeeffect_theo=ynodeeffect_theo,
            xedgeeffect_theo=xedgeeffect_theo,
            yedgeeffect_theo=yedgeeffect_theo,
        )

        for k in expected_theo.keys():
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    generated_theo[k], expected_theo[k]
                )
            )

    def test_education(self):
        """Checks coefficient matrices for direct, total and final effects of education example."""
        m, xdat = education()
        generated_theo = compute_theo_effects(m, xdat.mean(axis=1))

        # direct effects
        mx_theo = np.array(
            [
                [0.1, 0.1, -0.1, -0.5, "NaN", "NaN"],
                ["NaN", "NaN", "NaN", "NaN", "NaN", 1.0],
                ["NaN", "NaN", "NaN", "NaN", 1, "NaN"],
            ]
        ).astype(np.float64)
        my_theo = np.array(
            [["NaN", "NaN", "NaN"], [-1, "NaN", "NaN"], [1.0, 0.5, "NaN"]]
        ).astype(np.float64)

        # total effects
        ex_theo = np.array(
            [
                [0.1, 0.1, -0.1, -0.5, "NaN", "NaN"],
                [-0.1, -0.1, 0.1, 0.5, "NaN", 1],
                [0.05, 0.05, -0.05, -0.25, 1, 0.5],
            ]
        ).astype(np.float64)
        ey_theo = np.array([[1, "NaN", "NaN"], [-1, 1, "NaN"], [0.5, 0.5, 1]]).astype(
            np.float64
        )

        # final effects
        xnodeeffect_theo = np.array([0.05, 0.05, -0.05, -0.25, 1, 0.5]).astype(
            np.float64
        )
        ynodeeffect_theo = np.array([0.5, 0.5, 1]).astype(np.float64)
        xedgeeffect_theo = np.array(
            [
                [0.05, 0.05, -0.05, -0.25, "NaN", "NaN"],
                ["NaN", "NaN", "NaN", "NaN", "NaN", 0.5],
                ["NaN", "NaN", "NaN", "NaN", 1, "NaN"],
            ]
        ).astype(np.float64)
        yedgeeffect_theo = np.array(
            [["NaN", "NaN", "NaN"], [-0.5, "NaN", "NaN"], [1, 0.5, "NaN"]]
        ).astype(np.float64)

        expected_theo = dict(
            mx_theo=mx_theo,
            my_theo=my_theo,
            ex_theo=ex_theo,
            ey_theo=ey_theo,
            xnodeeffect_theo=xnodeeffect_theo,
            ynodeeffect_theo=ynodeeffect_theo,
            xedgeeffect_theo=xedgeeffect_theo,
            yedgeeffect_theo=yedgeeffect_theo,
        )
        for k in expected_theo.keys():
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    generated_theo[k], expected_theo[k]
                )
            )

    def test_example2_runs(self):
        """Test that example2 model runs without errors."""
        from causing.examples.models import example2

        m, xdat = example2()

        # Verify model structure
        self.assertEqual(len(m.xvars), 1)
        self.assertEqual(len(m.yvars), 1)

        # Verify computation works
        yhat = m.compute(xdat)
        self.assertEqual(yhat.shape[0], 1)  # 1 y variable

        # Verify effects calculation works
        effects = m.calc_effects(xdat)
        self.assertIn("yhat", effects)

    def test_example3_runs(self):
        """Test that example3 model runs without errors."""
        from causing.examples.models import example3

        m, xdat = example3()

        # Verify model structure
        self.assertEqual(len(m.xvars), 1)
        self.assertEqual(len(m.yvars), 3)

        # Verify computation works
        yhat = m.compute(xdat)
        self.assertEqual(yhat.shape[0], 3)  # 3 y variables

        # Verify effects calculation works
        effects = m.calc_effects(xdat)
        self.assertIn("yhat", effects)

    def test_heaviside_runs(self):
        """Test that heaviside model runs without errors."""
        from causing.examples.models import heaviside

        m, xdat = heaviside()

        # Verify model structure
        self.assertEqual(len(m.xvars), 1)
        self.assertEqual(len(m.yvars), 1)

        # Verify computation works
        yhat = m.compute(xdat)
        self.assertEqual(yhat.shape[0], 1)  # 1 y variable

        # Verify heaviside function behavior (Max(X1, 0))
        # xdat should have negative and positive values
        # Negative values should become 0, positive stay positive
        for i in range(xdat.shape[1]):
            expected = max(xdat[0, i], 0)
            self.assertAlmostEqual(yhat[0, i], expected)
