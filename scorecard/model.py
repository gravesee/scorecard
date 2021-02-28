# when a model is fit, need to record some information about it

from typing import Dict, List
from .variable import Variable
import copy
import numpy as np
import pandas as pd


def iter_coefs(coefs):
    for el in coefs:
        yield el


def zip_coefs_and_variables(
    coefs: np.ndarray, variables: Dict[str, Variable], step=[1]
):
    it = iter(coefs)  # type: ignore
    res = {}
    for name, v in variables.items():
        if v.step in step:
            labels = v.transform.labels
            res[name] = pd.Series({l: coef for l, coef in zip(labels, it)})
    return res


class Model:
    def __init__(self, step1, step2, variables: Dict[str, Variable], name: str = ""):
        self.step1 = copy.deepcopy(step1)
        self.step2 = copy.deepcopy(step2)
        self.variables = copy.deepcopy(variables)
        self.name = name
    
    def coefs(self, step=[1]):
        return zip_coefs_and_variables(self.step1.x, self.variables, step=step)
