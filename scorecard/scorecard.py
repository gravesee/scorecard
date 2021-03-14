import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack

from .carousel import Carousel
from .discretize import discretize
from .model import Model
from .performance import Performance
from .variable import Variable

EvalSets = Optional[List[Tuple[pd.DataFrame, Performance]]]

# TODO: figure out how to keep variables in sync across scorecard and models
# after model is registered, when collapsing or expanding variables, currently fit_info
# is being taken from the fitted models prior to the variables being altered. Need
# to somehow invalidated the fit info for the variable when it has been adjusted...
# or when adjustments are made need to set model to None....


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
            return cls(variables, df, perf)
        else:
            return cls(variables, None, None)

    def __init__(
        self,
        variables: Dict[str, Variable],
        df: Optional[pd.DataFrame] = None,
        perf: Optional[Performance] = None,
    ):
        self._model = Model(None, None, variables, name="init")
        self.variables = variables
        self.features = Carousel(list(variables.keys()))
        self._models: List[Model] = []  # list of all fitted models

        self.df = df
        self.perf = perf

    def fix_data(
        self, df: Optional[pd.DataFrame], perf: Optional[Performance]
    ) -> Tuple[pd.DataFrame, Performance]:
        if df is None and perf is None:
            return self.df, self.perf
        if df is None or perf is None:
            raise Exception("Must provide both `df` and `perf` or neither.")
        else:
            return df, perf

    def summary(
        self, df: Optional[pd.DataFrame] = None, perf: Optional[Performance] = None
    ):
        # create summary dataframe for each variable
        df, perf = self.fix_data(df, perf)

        res = []
        for v in self.variables.values():
            res.append(v.summary(df[v.name], perf))
        return pd.DataFrame.from_records(res)

    def to_sparse(
        self, df: Optional[pd.DataFrame] = None, step: List[Optional[int]] = [1],
    ):

        if df is None:
            df = self.df
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

        # constraints need to the offset of each sparse column index which is why they need to be
        # calculated in the scorecard rather than in the variable
        # TODO: move to variable and accept an offset term to keep track of column index
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
            df = self.df
        if self.model is None:
            raise Exception("no models have been fit yet")
        return self.model.predict(df)

    def _fit(
        self,
        df: Optional[pd.DataFrame] = None,
        perf: Optional[Performance] = None,
        offset: Optional[np.ndarray] = None,
        alpha: float = 0.001,
        step=[1],
    ):

        df, perf = self.fix_data(df, perf)

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
            step2, _ = self._fit(df, perf, preds, alpha, step=[2])

        self.save_model(
            Model(step1, step2, self.variables, f"model_{len(self._models):02d}")
        )

    def has_step_two_variables(self) -> bool:
        return any([v.step == 2 for v in self.variables.values()])

    def display_variable(
        self,
        var: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        perf: Optional[Performance] = None,
    ):
        if var is None:
            var = self.features.curr()

        df, perf = self.fix_data(df, perf)
        
        res = self[var].display(df[var], perf)

        fit_info = self.model.fit_info(self.variables[var])

        out = pd.concat([res, fit_info], axis=1)

        # add indices
        index, maxlen = [], out.index.str.len().max()
        for i, s in enumerate(out.index):
            index.append(f"[{i}] {s:>{maxlen}}")

        out.index = index

        # get the styles
        style = [*perf.style, *self.model.style]
        out = out.style.use(style)
        return out.set_na_rep('') 

    def next(self):
        self.features.next()
        return self.display_variable()

    def prev(self):
        self.features.prev()
        return self.display_variable()

    def collapse(self, l):
        self[self.features.curr()].collapse(l)

    def expand(self, i):
        self[self.features.curr()].expand(i)


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


def adjust(mod: Scorecard):
    import ipywidgets as widgets
    from IPython.display import display

    input_field = widgets.Text(description="Enter Command:", disabled=False)
    
    def render():
        var = mod.features.curr()
        with output:
            display(widgets.HTML(f"<b>{var}</b>Step: {mod.variables[var].step}"))
            display(mod.display_variable(var))
            output.clear_output(wait=True)

    def handle_command(command):
        var = mod.features.curr()
        if command in ["", "n"]:
            mod.features.next()
        elif command in ["p"]:
            mod.features.prev()
        elif command == "f":
            mod.fit()
        elif command == "s 1":
            mod[var].step = 1
        elif command == "s 2":
            mod[var].step = 2
        elif command == "- 1:2":
            mod[var].collapse([1, 2])

    def on_textbox_key_event(text):
        handle_command(text.value)
        input_field.value = ""
        with output:
            render()

    input_field.on_submit(on_textbox_key_event)

    output = widgets.Output()
    display(output, input_field)
