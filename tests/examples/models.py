import unittest
import numpy as np

from causing.examples.models import example, education


class TestExampleModels(unittest.TestCase):
    def test_example(self):
        """Checks coefficient matrices for direct, total and final effects of example."""
        m, xdat, _, _ = example()
        generated_theo = m.theo(xdat.mean(axis=1))

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
        exj_theo = np.array([12.92914837, 1]).astype(np.float64)
        eyj_theo = np.array([12.92914837, 1, 1]).astype(np.float64)
        eyx_theo = np.array([[12.92914837, "NaN"], ["NaN", 1], ["NaN", "NaN"]]).astype(
            np.float64
        )
        eyy_theo = np.array(
            [["NaN", "NaN", "NaN"], [11.92914837, "NaN", "NaN"], [1, 1, "NaN"]]
        ).astype(np.float64)

        expected_theo = dict(
            mx_theo=mx_theo,
            my_theo=my_theo,
            ex_theo=ex_theo,
            ey_theo=ey_theo,
            exj_theo=exj_theo,
            eyj_theo=eyj_theo,
            eyx_theo=eyx_theo,
            eyy_theo=eyy_theo,
        )

        for k in expected_theo.keys():
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    generated_theo[k], expected_theo[k]
                )
            )

    def test_education(self):
        """Checks coefficient matrices for direct, total and final effects of education example."""
        m, xdat, _, _ = education()
        generated_theo = m.theo(xdat.mean(axis=1))

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
        exj_theo = np.array([0.05, 0.05, -0.05, -0.25, 1, 0.5]).astype(np.float64)
        eyj_theo = np.array([0.5, 0.5, 1]).astype(np.float64)
        eyx_theo = np.array(
            [
                [0.05, 0.05, -0.05, -0.25, "NaN", "NaN"],
                ["NaN", "NaN", "NaN", "NaN", "NaN", 0.5],
                ["NaN", "NaN", "NaN", "NaN", 1, "NaN"],
            ]
        ).astype(np.float64)
        eyy_theo = np.array(
            [["NaN", "NaN", "NaN"], [-0.5, "NaN", "NaN"], [1, 0.5, "NaN"]]
        ).astype(np.float64)

        expected_theo = dict(
            mx_theo=mx_theo,
            my_theo=my_theo,
            ex_theo=ex_theo,
            ey_theo=ey_theo,
            exj_theo=exj_theo,
            eyj_theo=eyj_theo,
            eyx_theo=eyx_theo,
            eyy_theo=eyy_theo,
        )
        for k in expected_theo.keys():
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    generated_theo[k], expected_theo[k]
                )
            )
