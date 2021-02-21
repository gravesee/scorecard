from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier

from .performance import BinaryPerformance, Performance
from .transform import *
from .variable import Variable


def binary_discretizer(
    x: pd.Series, perf: Performance, exceptions: List[float], **kwargs
):
    y, w = perf
    f = ~np.isnan(x) & ~np.isnan(y) & ~np.isin(x, exceptions)  # type: ignore
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(x[f].values.reshape(-1, 1), y[f], sample_weight=w[f])
    res = [v for v, x in zip(clf.tree_.threshold, clf.tree_.feature) if x != -2]
    return res


# TODO: register discretizers with the performance class
# TODO: check vectors are all the same length?


def discretize(
    df: pd.DataFrame,
    perf: Performance,
    missing: float = np.nan,
    exceptions: Optional[Union[Dict[str, List[Any]], List[Any]]] = None,
    discretizer: str = None,
    **kwargs,
):

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
            breaks = binary_discretizer(df[col], perf, excepts, **kwargs)
            tf = ContinuousTransform(breaks, excepts, missing)
        else:
            levels = []
            excepts = exceptions.get(col, [])
            for el in list(df[col].unique()):  # type: ignore
                if el not in (excepts + [missing]):
                    levels.append(el)
            tf = CategoricalTransform(levels, excepts, missing)

        res[col] = Variable(df[col], perf, tf)

    return res
