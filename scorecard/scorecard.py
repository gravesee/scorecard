from typing import Any, Dict, List, Optional, Union

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize

from .discretize import discretize
from .model import Model
from .performance import Performance
from .variable import Variable


class Scorecard:
    @classmethod
    def discretize(
        cls,
        df: pd.DataFrame,
        perf: Performance,
        missing: float = np.nan,
        exceptions: Optional[Union[Dict[str, List[Any]], List[Any]]] = None,
        **kwargs,
    ):
        variables = discretize(df, perf, missing, exceptions, **kwargs)
        return cls(variables)

    def __init__(self, variables: Dict[str, Variable]):
        self.variables = variables
        self._models = []

    def summary(self, df: pd.DataFrame, perf: Performance):
        # create summary dataframe for each variable
        res = []
        for v in self.variables.values():
            res.append(v.summary(df[v.name], perf))
        return pd.DataFrame.from_records(res)

    def to_sparse(
        self,
        df: pd.DataFrame,
        step: List[Optional[int]] = [1],
    ):
        res = []
        for k, v in self.variables.items():
            if v.step in step:
                res.append(v.to_sparse(df[k]))
        M = hstack(res)
        return hstack([M, np.ones(M.shape[0]).reshape(-1, 1)])

    def __getitem__(self, key):
        return self.variables[key]

    def predict(self, df: pd.DataFrame):
        if len(self._models) == 0:
            raise Exception("no models have been fit yet")

        obj = self._models[-1]
        M = self.to_sparse(df)

        return M @ obj.obj.x

    def fit(
        self,
        df: pd.DataFrame,
        perf: Performance,
        offset: Optional[np.ndarray] = None,
        alpha: float = 0.001,
    ):
        M = self.to_sparse(df)

        if offset is None:
            offset = np.zeros(M.shape[0])

        y, w = perf

        coefs = np.zeros(M.shape[1])
        obj = minimize(
            logistic_loss,
            coefs,
            (M, y.values, w.values, offset, alpha),
            jac=logistic_gradient,
        )

        self._models.append(
            Model(obj, self.variables, f"model_{len(self._models):02d}")
        )


def sigmoid(x):
    return ne.evaluate("1 / (1 + exp(-x))")


def logistic_loss(coefs, X, y, w, offset, alpha=0.001) -> float:
    """w is always provided as an array of 1s at the very least"""

    h = sigmoid((X @ coefs) + offset)
    m = ne.evaluate("sum(w)")

    weighted_cost = ne.evaluate("(y * log(h) + (1 - y) * log(1 - h)) * w")
    cost = -np.sum(weighted_cost) / m

    # regularization
    reg = (alpha * np.sum(coefs[1:] ** 2)) / 2
    return cost + reg


def logistic_gradient(coefs, X, y, w, offset, alpha=0.001) -> float:
    h = sigmoid((X @ coefs) + offset)

    grads = X.T @ (w * (h - y))  # multiple error by weight

    # ignore intercept during update
    grads[:-1] = grads[:-1] + alpha * grads[:-1]

    return grads / w.sum()
