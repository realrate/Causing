import json
from math import floor, log10

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
