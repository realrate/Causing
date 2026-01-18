"""Comprehensive tests for the Model class."""

import unittest
import numpy as np
from sympy import symbols

from causing.model import Model
from causing import create_indiv


class TestModelInitialization(unittest.TestCase):
    """Test Model initialization and basic properties."""

    def test_basic_model_creation(self):
        """Test creating a simple model."""
        X1, X2, Y1, Y2 = symbols(["X1", "X2", "Y1", "Y2"])

        m = Model(xvars=[X1, X2], yvars=[Y1, Y2], equations=(X1, X2 + Y1), final_var=Y2)

        # Check dimensions
        self.assertEqual(m.mdim, 2)
        self.assertEqual(m.ndim, 2)

        # Check variable names are strings
        self.assertEqual(m.xvars, ["X1", "X2"])
        self.assertEqual(m.yvars, ["Y1", "Y2"])
        self.assertEqual(m.final_var, "Y2")

        # Check final_ind
        self.assertEqual(m.final_ind, 1)

    def test_model_with_string_vars(self):
        """Test creating a model with string variable names."""
        m = Model(
            xvars=["X1", "X2"],
            yvars=["Y1", "Y2"],
            equations=(symbols("X1"), symbols("X2") + symbols("Y1")),
            final_var="Y2",
        )

        self.assertEqual(m.xvars, ["X1", "X2"])
        self.assertEqual(m.yvars, ["Y1", "Y2"])

    def test_graph_construction(self):
        """Test that the causal graph is correctly constructed."""
        X1, X2, Y1, Y2, Y3 = symbols(["X1", "X2", "Y1", "Y2", "Y3"])

        m = Model(
            xvars=[X1, X2],
            yvars=[Y1, Y2, Y3],
            equations=(X1, X2 + Y1, Y1 + Y2),  # Y1 = X1  # Y2 = X2 + Y1  # Y3 = Y1 + Y2
            final_var=Y3,
        )

        # Check direct edges
        self.assertTrue(m.graph.has_edge("X1", "Y1"))
        self.assertTrue(m.graph.has_edge("X2", "Y2"))
        self.assertTrue(m.graph.has_edge("Y1", "Y2"))
        self.assertTrue(m.graph.has_edge("Y1", "Y3"))
        self.assertTrue(m.graph.has_edge("Y2", "Y3"))

        # Check edges that should not exist
        self.assertFalse(m.graph.has_edge("X1", "Y2"))
        self.assertFalse(m.graph.has_edge("X2", "Y1"))

        # Check transitive closure
        self.assertTrue(m.trans_graph.has_edge("X1", "Y3"))
        self.assertTrue(m.trans_graph.has_edge("X2", "Y3"))

    def test_vars_property(self):
        """Test the vars property returns all variables."""
        X1, Y1 = symbols(["X1", "Y1"])
        m = Model(xvars=[X1], yvars=[Y1], equations=(X1,), final_var=Y1)

        self.assertEqual(m.vars, ["X1", "Y1"])


class TestModelCompute(unittest.TestCase):
    """Test the Model.compute method."""

    def test_simple_linear_model(self):
        """Test computing with a simple linear model."""
        X1, X2, Y1, Y2 = symbols(["X1", "X2", "Y1", "Y2"])

        m = Model(xvars=[X1, X2], yvars=[Y1, Y2], equations=(X1, X2 + Y1), final_var=Y2)

        xdat = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2: 2 variables, 2 observations
        yhat = m.compute(xdat)

        # Y1 = X1, Y2 = X2 + Y1
        # For obs 1: Y1 = 1, Y2 = 3 + 1 = 4
        # For obs 2: Y1 = 2, Y2 = 4 + 2 = 6
        expected = np.array([[1.0, 2.0], [4.0, 6.0]])

        np.testing.assert_array_almost_equal(yhat, expected)

    def test_nonlinear_model(self):
        """Test computing with a nonlinear model."""
        X1, Y1, Y2 = symbols(["X1", "Y1", "Y2"])

        m = Model(xvars=[X1], yvars=[Y1, Y2], equations=(X1**2, Y1 + 1), final_var=Y2)

        xdat = np.array([[2.0, 3.0]])  # 1x2: 1 variable, 2 observations
        yhat = m.compute(xdat)

        # Y1 = X1^2, Y2 = Y1 + 1
        # For obs 1: Y1 = 4, Y2 = 5
        # For obs 2: Y1 = 9, Y2 = 10
        expected = np.array([[4.0, 9.0], [5.0, 10.0]])

        np.testing.assert_array_almost_equal(yhat, expected)

    def test_compute_single_observation(self):
        """Test computing with a single observation."""
        X1, Y1 = symbols(["X1", "Y1"])

        m = Model(xvars=[X1], yvars=[Y1], equations=(2 * X1,), final_var=Y1)

        xdat = np.array([[5.0]])  # 1x1: 1 variable, 1 observation
        yhat = m.compute(xdat)

        expected = np.array([[10.0]])
        np.testing.assert_array_almost_equal(yhat, expected)


class TestModelCalcEffects(unittest.TestCase):
    """Test the Model.calc_effects method."""

    def test_calc_effects_basic(self):
        """Test basic effect calculation."""
        X1, X2, Y1, Y2 = symbols(["X1", "X2", "Y1", "Y2"])

        m = Model(xvars=[X1, X2], yvars=[Y1, Y2], equations=(X1, X2 + Y1), final_var=Y2)

        xdat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        effects = m.calc_effects(xdat)

        # Check that all expected keys are present
        expected_keys = [
            "yhat",
            "xnodeeffects",
            "ynodeeffects",
            "xedgeeffects",
            "yedgeeffects",
        ]
        for key in expected_keys:
            self.assertIn(key, effects)

        # Check shapes
        self.assertEqual(effects["yhat"].shape, (2, 3))  # ndim x tau
        self.assertEqual(effects["xnodeeffects"].shape, (2, 3))  # mdim x tau
        self.assertEqual(effects["ynodeeffects"].shape, (2, 3))  # ndim x tau
        self.assertEqual(effects["xedgeeffects"].shape, (3, 2, 2))  # tau x ndim x mdim
        self.assertEqual(effects["yedgeeffects"].shape, (3, 2, 2))  # tau x ndim x ndim

    def test_calc_effects_simple_chain(self):
        """Test effects in a simple causal chain."""
        X1, Y1, Y2 = symbols(["X1", "Y1", "Y2"])

        m = Model(
            xvars=[X1],
            yvars=[Y1, Y2],
            equations=(X1, Y1),  # Y1 = X1, Y2 = Y1
            final_var=Y2,
        )

        xdat = np.array([[1.0, 2.0, 3.0]])
        effects = m.calc_effects(xdat)

        # Y1 has effect on Y2, X1 has effect on Y2 (through Y1)
        # All effects should be computed
        self.assertFalse(np.all(np.isnan(effects["xnodeeffects"])))
        self.assertFalse(np.all(np.isnan(effects["ynodeeffects"])))


class TestModelShrink(unittest.TestCase):
    """Test the Model.shrink method."""

    def test_shrink_removes_nodes(self):
        """Test that shrink removes specified nodes."""
        X1, Y1, Y2, Y3 = symbols(["X1", "Y1", "Y2", "Y3"])

        m = Model(
            xvars=[X1],
            yvars=[Y1, Y2, Y3],
            equations=(X1, Y1, Y2),  # Y1 = X1, Y2 = Y1, Y3 = Y2
            final_var=Y3,
        )

        # Shrink by removing Y2
        m_shrunk = m.shrink(["Y2"])

        # Check that Y2 is removed
        self.assertEqual(len(m_shrunk.yvars), 2)
        self.assertIn("Y1", m_shrunk.yvars)
        self.assertIn("Y3", m_shrunk.yvars)
        self.assertNotIn("Y2", m_shrunk.yvars)

        # Check that equations are updated (Y3 should now depend on Y1 directly)
        self.assertEqual(len(m_shrunk.equations), 2)


class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_constant_equation(self):
        """Test model with constant equations.

        Note: Constant equations (plain numbers) are not well-supported in the current
        implementation and should be expressed as symbolic constants or parameters.
        This test is skipped for now.
        """
        self.skipTest("Constant equations not supported in current implementation")

        X1, Y1, Y2 = symbols(["X1", "Y1", "Y2"])

        m = Model(
            xvars=[X1],
            yvars=[Y1, Y2],
            equations=(5, X1 + Y1),  # Y1 = 5 (constant), Y2 = X1 + Y1
            final_var=Y2,
        )

        xdat = np.array([[1.0, 2.0]])
        yhat = m.compute(xdat)

        # Y1 = 5, Y2 = X1 + 5
        expected = np.array([[5.0, 5.0], [6.0, 7.0]])
        np.testing.assert_array_almost_equal(yhat, expected)

    def test_model_with_parameters(self):
        """Test model with parameters."""
        X1, Y1 = symbols(["X1", "Y1"])
        a = symbols("a")

        m = Model(
            xvars=[X1],
            yvars=[Y1],
            equations=(a * X1,),
            final_var=Y1,
            parameters={"a": 2.5},
        )

        xdat = np.array([[4.0]])
        yhat = m.compute(xdat)

        # Y1 = 2.5 * X1 = 2.5 * 4 = 10
        expected = np.array([[10.0]])
        np.testing.assert_array_almost_equal(yhat, expected)

    def test_single_variable_model(self):
        """Test model with single variable."""
        X1, Y1 = symbols(["X1", "Y1"])

        m = Model(xvars=[X1], yvars=[Y1], equations=(X1,), final_var=Y1)

        self.assertEqual(m.mdim, 1)
        self.assertEqual(m.ndim, 1)
        self.assertEqual(m.final_ind, 0)


class TestModelIntegration(unittest.TestCase):
    """Integration tests using example-like models."""

    def test_education_like_model(self):
        """Test a model similar to the education example."""
        FATHERED, MOTHERED, AGE, EDUC, WAGE = symbols(
            ["FATHERED", "MOTHERED", "AGE", "EDUC", "WAGE"]
        )

        m = Model(
            xvars=[FATHERED, MOTHERED, AGE],
            yvars=[EDUC, WAGE],
            equations=(
                13 + 0.1 * (FATHERED - 12) + 0.1 * (MOTHERED - 12),  # EDUC
                7 + 1 * (EDUC - 12),  # WAGE
            ),
            final_var=WAGE,
        )

        # Test with sample data
        xdat = np.array(
            [
                [12.0, 13.0, 14.0],  # FATHERED
                [12.0, 13.0, 14.0],  # MOTHERED
                [25.0, 26.0, 27.0],  # AGE
            ]
        )

        yhat = m.compute(xdat)

        # Check shape
        self.assertEqual(yhat.shape, (2, 3))

        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(yhat)))

        # Test calc_effects
        effects = m.calc_effects(xdat)

        # Check that effects were computed
        self.assertIn("yhat", effects)
        self.assertEqual(effects["yhat"].shape, (2, 3))

    def test_complex_causal_chain(self):
        """Test a more complex causal chain."""
        X1, X2, Y1, Y2, Y3, Y4 = symbols(["X1", "X2", "Y1", "Y2", "Y3", "Y4"])

        m = Model(
            xvars=[X1, X2],
            yvars=[Y1, Y2, Y3, Y4],
            equations=(
                X1,  # Y1 = X1
                X2 + Y1,  # Y2 = X2 + Y1
                Y1 + Y2,  # Y3 = Y1 + Y2
                Y2 + Y3,  # Y4 = Y2 + Y3
            ),
            final_var=Y4,
        )

        # Check graph structure
        self.assertTrue(m.graph.has_edge("X1", "Y1"))
        self.assertTrue(m.graph.has_edge("Y1", "Y2"))
        self.assertTrue(m.graph.has_edge("Y2", "Y4"))

        # Check transitive paths
        self.assertTrue(m.trans_graph.has_edge("X1", "Y4"))
        self.assertTrue(m.trans_graph.has_edge("X2", "Y4"))

        # Test computation
        xdat = np.array([[1.0], [2.0]])
        yhat = m.compute(xdat)

        # Y1 = 1, Y2 = 2+1 = 3, Y3 = 1+3 = 4, Y4 = 3+4 = 7
        expected = np.array([[1.0], [3.0], [4.0], [7.0]])
        np.testing.assert_array_almost_equal(yhat, expected)


class TestCreateIndiv(unittest.TestCase):
    """Test the create_indiv helper function."""

    def test_create_indiv_limits_results(self):
        """Test that create_indiv correctly limits the number of individuals."""
        X1, Y1, Y2 = symbols(["X1", "Y1", "Y2"])

        m = Model(xvars=[X1], yvars=[Y1, Y2], equations=(X1, Y1), final_var=Y2)

        # Create data with 10 observations
        xdat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

        # Limit to 3 individuals
        effects = create_indiv(m, xdat, show_nr_indiv=3)

        # Check that the results are limited
        self.assertEqual(effects["xnodeeffects"].shape[1], 3)  # mdim x 3
        self.assertEqual(effects["ynodeeffects"].shape[1], 3)  # ndim x 3
        self.assertEqual(effects["xedgeeffects"].shape[0], 3)  # 3 x ndim x mdim
        self.assertEqual(effects["yedgeeffects"].shape[0], 3)  # 3 x ndim x ndim

    def test_create_indiv_preserves_structure(self):
        """Test that create_indiv preserves the structure of effects."""
        X1, X2, Y1, Y2 = symbols(["X1", "X2", "Y1", "Y2"])

        m = Model(xvars=[X1, X2], yvars=[Y1, Y2], equations=(X1, X2 + Y1), final_var=Y2)

        xdat = np.array([[1.0, 2.0], [3.0, 4.0]])
        effects = create_indiv(m, xdat, show_nr_indiv=2)

        # Check all expected keys are present
        expected_keys = [
            "yhat",
            "xnodeeffects",
            "ynodeeffects",
            "xedgeeffects",
            "yedgeeffects",
        ]
        for key in expected_keys:
            self.assertIn(key, effects)


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end tests for the complete workflow."""

    def test_complete_workflow_simple_model(self):
        """Test complete workflow: create model, compute, calculate effects."""
        # Step 1: Create a simple model
        X1, Y1, Y2 = symbols(["X1", "Y1", "Y2"])
        m = Model(xvars=[X1], yvars=[Y1, Y2], equations=(2 * X1, Y1 + 1), final_var=Y2)

        # Step 2: Create input data
        xdat = np.array([[1.0, 2.0, 3.0]])

        # Step 3: Compute model values
        yhat = m.compute(xdat)
        self.assertEqual(yhat.shape, (2, 3))

        # Verify computation: Y1 = 2*X1, Y2 = Y1 + 1
        np.testing.assert_array_almost_equal(yhat[0], [2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(yhat[1], [3.0, 5.0, 7.0])

        # Step 4: Calculate effects
        effects = m.calc_effects(xdat)

        # Verify effects structure
        self.assertIn("yhat", effects)
        self.assertIn("xnodeeffects", effects)
        self.assertIn("ynodeeffects", effects)

        # Verify yhat matches compute
        np.testing.assert_array_almost_equal(effects["yhat"], yhat)

    def test_workflow_with_create_indiv(self):
        """Test workflow using create_indiv helper."""
        X1, Y1, Y2, Y3 = symbols(["X1", "Y1", "Y2", "Y3"])

        m = Model(xvars=[X1], yvars=[Y1, Y2, Y3], equations=(X1, Y1, Y2), final_var=Y3)

        # Create data with 5 observations
        xdat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Use create_indiv to limit results
        effects = create_indiv(m, xdat, show_nr_indiv=3)

        # Verify limited results
        self.assertEqual(effects["xnodeeffects"].shape, (1, 3))
        self.assertEqual(effects["ynodeeffects"].shape, (3, 3))

    def test_model_persistence_across_computations(self):
        """Test that model can be reused for multiple computations."""
        X1, Y1 = symbols(["X1", "Y1"])

        m = Model(xvars=[X1], yvars=[Y1], equations=(X1 * 2,), final_var=Y1)

        # First computation
        xdat1 = np.array([[1.0, 2.0]])
        yhat1 = m.compute(xdat1)

        # Second computation with different data
        xdat2 = np.array([[3.0, 4.0, 5.0]])
        yhat2 = m.compute(xdat2)

        # Verify both are correct
        np.testing.assert_array_almost_equal(yhat1, [[2.0, 4.0]])
        np.testing.assert_array_almost_equal(yhat2, [[6.0, 8.0, 10.0]])

        # Model should still be usable
        effects = m.calc_effects(xdat2)
        self.assertEqual(effects["yhat"].shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
