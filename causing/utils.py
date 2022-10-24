import json
import math
from math import floor, log10
import locale

import numpy as np


@np.vectorize
def round_sig(x, sig=2) -> float:
    """Round x to the given number of significant figures"""
    if x == 0 or not np.isfinite(x):
        return x
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def round_sig_recursive(x, sig=2):
    """Round all floats in x to the given number of significant figures

    x can be a nested data structure.
    """
    if isinstance(x, dict):
        return {key: round_sig_recursive(value, sig) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return x.__class__(round_sig_recursive(value, sig) for value in x)
    if isinstance(x, (float, np.ndarray)):
        return round_sig(x, sig)
    # avoid importing pytorch for isinstance check
    if type(x).__name__ == "Tensor":
        return x.apply_(lambda x: round_sig(x, sig))
    # avoid importing pandas for isinstance check
    if type(x).__name__ == "DataFrame":
        return x.apply(lambda x: round_sig(x, sig))

    return x


class MatrixEncoder(json.JSONEncoder):
    def default(self, obj):
        # allow serialization of numpy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # avoid importing pytorch for isinstance check
        if isinstance(obj, np.ndarray) or type(obj).__name__ == "Tensor":
            return obj.tolist()
        # avoid importing pandas for isinstance check
        if type(obj).__name__ == "DataFrame":
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def dump_json(data, filename, allow_nan=True):
    with open(filename, "w") as f:
        json.dump(
            data, f, sort_keys=True, indent=4, cls=MatrixEncoder, allow_nan=allow_nan
        )
        f.write("\n")


def fmt_min_sig(x, min_sig_figures=3, percent=False, percent_spacer=""):
    """Format number with at least the given amount of significant figures.
    See https://www.karl.berlin/formatting-numbers.html
    """
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        num = "0"
    else:
        if percent:
            x *= 100
        show_dec = max(-math.floor(math.log10(abs(x)) + 1) + min_sig_figures, 0)
        num = locale.format_string("%." + str(show_dec) + "f", x, grouping=True)
    if percent:
        num += percent_spacer + "%"
    return num
