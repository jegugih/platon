import pickle
import weakref
from pathlib import Path

import numpy as np

# File I/O stays in NumPy; arrays are converted to MLX by the callers.
# Large read-only data is shared between calculators (see get_shared).


def load_dict_from_pickle(filename):
    basedir = Path(__file__).resolve().parent
    with open(basedir / filename, "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = np.array(dictionary[key])

        return dictionary


def load_numpy(filename):
    basedir = Path(__file__).resolve().parent
    return np.load(basedir / filename)


class _SharedValue:
    def __init__(self, value):
        self.value = value


_shared_values = weakref.WeakValueDictionary()


def get_shared(key, load):
    '''Returns a holder whose .value is the read-only data identified by
    `key`, calling load() only if no live calculator already holds it.  The
    data is freed once nothing references the holder, so keep a reference to
    the holder (not just its value) for as long as the data is used.'''
    holder = _shared_values.get(key)
    if holder is None:
        holder = _SharedValue(load())
        _shared_values[key] = holder
    return holder
