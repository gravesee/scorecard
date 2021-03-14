# when a model is fit, need to record some information about it

from typing import Dict, List
from .variable import Variable
import copy
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from pandas.io.formats.style import Styler


def iter_coefs(coefs):
    for el in coefs:
        yield el


def zip_coefs_and_variables(coefs: np.ndarray, variables: Dict[str, Variable], step=1):
    it = iter(coefs)  # type: ignore
    res = {}
    for name, v in variables.items():
        if v.step == step:
            labels = v.transform.labels
            res[name] = pd.Series({l: coef for l, coef in zip(labels, it)})
    return res


class Model:
    def __init__(self, step1, step2, variables: Dict[str, Variable], name: str = ""):
        self.step1 = copy.deepcopy(step1)
        self.step2 = copy.deepcopy(step2)
        self.variables = copy.deepcopy(variables)
        self.name = name

        s = Styler(pd.DataFrame())
        s = s.bar(subset=['Preds'], align='zero', color='red')
        s = s.bar(subset=['Step2'], align='zero', color='blue')
        self._style = s.export()

    def coefs(self, step=1):
        coefs = self.step1.x if step == 1 else self.step2.x
        return zip_coefs_and_variables(coefs, self.variables, step=step)

    def to_categorical(self, df: pd.DataFrame, step=1):
        res = []
        for k, v in self.variables.items():
            if v.step == step:
                res.append(v.to_sparse(df[k]))
        M = hstack(res)
        return hstack([M, np.ones(M.shape[0]).reshape(-1, 1)])

    def predict(self, df: pd.DataFrame, step=1):
        obj = self.step1 if step == 1 else self.step2
        M = self.to_categorical(df, step=step)
        return M @ obj.x

    def fit_info(self, var: Variable):
        ix = self.variables[var.name].transform.labels
        res = pd.DataFrame(index=ix, columns=["Preds", "Step2"], dtype=float)

        # check if current variable hash equals the model variable hash,
        # if so, then the fit_info can be applied, otherwise it should be empty
        if hash(var) != hash(self.variables[var.name]):
            return res

        if self.step1 is not None:
            res["Preds"] = self.coefs(1).get(var.name, None)

        if self.step2 is not None:
            res["Step2"] = self.coefs(2).get(var.name, None)

        return res.fillna(0)
    
    @property
    def style(self):
        return self._style
