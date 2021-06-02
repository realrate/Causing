import unittest

from causing.utils import round_sig_recursive


class TestRoundSigRecursive(unittest.TestCase):
    def test_recursive(self) -> None:
        orig = {
            "a_list": [111.0, 0.111],
            "a_tuple": (111.0, 0.111),
            "a_dict": {"a": 111.0, "b": 0.111},
        }
        rounded = {
            "a_list": [100, 0.1],
            "a_tuple": (100, 0.1),
            "a_dict": {"a": 100, "b": 0.1},
        }
        self.assertEqual(round_sig_recursive(orig, 1), rounded)
