from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier

from .performance import BinaryPerformance
from .transform import *


def binary_discretizer(x, y, w, exceptions, **kwargs):
    f = ~np.isnan(x) & ~np.isnan(y) & ~np.isin(x, exceptions)  # type: ignore
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(x[f].values.reshape(-1, 1), y[f], sample_weight=w[f])
    tree = clf.tree_
    res = [v for v, x in zip(tree.threshold, tree.feature) if x != -2]
    return res


def discretize(
    df: pd.DataFrame,
    y: pd.Series,
    w: Optional[pd.Series] = None,
    missing: float = np.nan,
    exceptions: Optional[Union[Dict[str, List[Any]], List[Any]]] = None,
    discretizer: str = None,
    performance: str = "infer",
    **kwargs
):
    if np.all(np.isin(y, [0, 1, np.nan])):
        performance = "binary"
    else:
        raise Exception("cannot infer performance from `y`")
    
    if w is None:
        w = pd.Series(np.ones_like(y))

    if exceptions is None:
        exceptions = {k: [] for k in df.columns}
    elif isinstance(exceptions, list):
        exceptions = {k: exceptions for k in df.columns}
    else:
        cols = set(exceptions.keys()) - set(df.columns)
        if len(cols) > 0:
            raise Exception(
                f"exceptions dict contains columns not found in `df`: {', '.join(cols)}"
            )

    res = {}
    for col in df.columns:
        # hard-code to binary for now
        if is_numeric_dtype(df[col]):
            excepts = exceptions.get(col, [])
            breaks = binary_discretizer(df[col], y, w, excepts, **kwargs)
            tf = ContinuousTransform(breaks, excepts, missing)
        else:
            levels = []
            excepts = exceptions.get(col, [])
            for el in list(df[col].unique()):  # type: ignore
                if el not in (excepts + [missing]):
                    levels.append(el)
            tf = CategoricalTransform(levels, excepts, missing)

        res[col] = tf

    return res
