from typing import Any, Dict, List, Optional, Union
import copy

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
        self._model = Model(None, variables, name="init")
        self.variables = variables
        self._models = []  # list of all fitted models

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

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, mod: Model):
        self._model = mod

    @property
    def models(self):
        return [m.name for m in self._models]

    def get_constraints(self, step=[1]):
        """get constraints for SLQSP optimizer"""
        # {"type": type_, "indices": indices, "len": len(labels)}
        i, res = 0, []
        for v in self.variables.values():
            if v.step in step:
                n, constraints = v.get_constraints()
                for constr in constraints:
                    # create constraint dict for SLQSP optimizer
                    base, target = constr["indices"]
                    type_ = constr["type"]

                    if type_ == "neu":
                        fun = eval(f"lambda x: x[{base + i}]")
                        res.append({"type": "eq", "fun": fun})
                    else:
                        fun = eval(f"lambda x: x[{base + i}] - x[{target + i}]")
                        if type_ == "=":
                            res.append({"type": "eq", "fun": fun})
                        else:
                            res.append({"type": "ineq", "fun": fun})
                i += n
        
        return res

    def save_model(self, mod: Model):
        self._models.append(mod)
        self.model = mod

    def load_model(self, model_id: Union[str, int]):
        """load fitted model and variables as they existed when the selected model was fit"""
        if isinstance(model_id, str):
            model = None
            for m in self._models:
                if m.name == model_id:
                    model = m
            if model is not None:
                self.model = model
                self.variables = copy.deepcopy(model.variables)
            else:
                raise Exception(f"no model found with name: {model_id}")
        elif isinstance(model_id, int):
            if (model_id < len(self._models) - 1) and (model_id > 0):
                model = self._models[model_id]
                self.model = model
                self.variables = copy.deepcopy(model.variables)
            else:
                raise Exception(f"invalid model index: {model_id}")
        else:
            raise Exception("model_id must be model name or a model index")

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise Exception("no models have been fit yet")
        M = self.to_sparse(df)
        return M @ self.model.coefs

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

        constraints = self.get_constraints()
        
        coefs = np.zeros(M.shape[1])
        
        obj = minimize(
            logistic_loss,
            coefs,
            (M, y.values, w.values, offset, alpha),
            jac=logistic_gradient,
            method="SLSQP",
            constraints=constraints,
        )

        self.save_model(Model(obj, self.variables, f"model_{len(self._models):02d}"))


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
