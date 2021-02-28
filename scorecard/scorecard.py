from typing import Any, Dict, List, Optional, Union, Tuple
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

EvalSets = Optional[List[Tuple[pd.DataFrame, Performance]]]

class Scorecard:
    @classmethod
    def discretize(
        cls,
        df: pd.DataFrame,
        perf: Performance,
        missing: float = np.nan,
        exceptions: Optional[Union[Dict[str, List[Any]], List[Any]]] = None,
        keep_data: bool = True,
        **kwargs,
    ):
        variables = discretize(df, perf, missing, exceptions, **kwargs)
        if keep_data:
            eval_sets = [(df, perf)]
        else:
            eval_sets = None
        return cls(variables, eval_sets)

    def __init__(self, variables: Dict[str, Variable], eval_sets = None):
        self._model = Model(None, None, variables, name="init")
        self.variables = variables
        self._models = []  # list of all fitted models

        self.eval_sets: EvalSets = eval_sets
    
    @property
    def training_data(self):
        if self.eval_sets is None:
            raise Exception("no eval_sets registered with Scorecard.")
        return self.eval_sets[0]

    def summary(self, df: Optional[pd.DataFrame] = None, perf: Optional[Performance] = None):
        # create summary dataframe for each variable
        if df is None:
            df, perf = self.training_data
        
        res = []
        for v in self.variables.values():
            res.append(v.summary(df[v.name], perf))
        return pd.DataFrame.from_records(res)

    def to_sparse(
        self,
        df: Optional[pd.DataFrame] = None,
        step: List[Optional[int]] = [1],
    ):
        if df is None:
            df, _ = self.training_data
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

    def predict(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df, _ = self.training_data
        if self.model is None:
            raise Exception("no models have been fit yet")
        M = self.to_sparse(df)
        return M @ self.model.step1.x

    def _fit(
        self,
        df: Optional[pd.DataFrame] = None,
        perf: Optional[Performance] = None,
        offset: Optional[np.ndarray] = None,
        alpha: float = 0.001,
        step = [1],
    ):

        if df is None or perf is None:
            df, perf = self.training_data

        M = self.to_sparse(df, step=step)

        if offset is None:
            offset = np.zeros(M.shape[0])

        y, w = perf

        constraints = self.get_constraints(step=step)
        
        coefs = np.zeros(M.shape[1])
        
        obj = minimize(
            logistic_loss,
            coefs,
            (M, y.values, w.values, offset, alpha),
            jac=logistic_gradient,
            method="SLSQP",
            constraints=constraints,
        )

        preds = M @ obj.x
        
        return obj, preds

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        perf: Optional[Performance] = None,
        offset: Optional[np.ndarray] = None,
        alpha: float = 0.001,
    ):


        step1, preds = self._fit(df, perf, offset, alpha, step=[1])

        step2 = None
        if self.has_step_two_variables():
            step2, _ = self._fit(df, perf, offset, alpha, step=[2])
        
        # TODO: call fit again, passing in an offset and step [2]
        # opportunity to re-use the exact same function 
        # TODO: refactor to _fit and fit
        # _fit can be re-used, fit calls _fit twice and saves the model object

        self.save_model(Model(step1, step2, self.variables, f"model_{len(self._models):02d}"))
    
    def has_step_two_variables(self) -> bool:
        return any([v.step == 2 for v in self.variables.values()])
    
    def display_variable(self, var: str, df: Optional[pd.DataFrame] = None, perf: Optional[Performance] = None):
        if df is None or perf is None:
            eval_sets = self.eval_sets
        else:
            eval_sets = [(df, perf)]
        
        res = []
        for df, perf in eval_sets:
            res.append(self[var].display(df[var], perf))
        
        # add model fit information
        coefs = self.model.coefs(step=[1]).get(var, None)
        if coefs is not None:
            return pd.concat([*res, coefs.rename("Preds")], axis=1)
        else:
            return pd.concat(res, axis=1)
    
    
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
