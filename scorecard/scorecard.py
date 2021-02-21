from typing import Dict, Optional, Union, List, Any
from scorecard.performance import Performance
from .variable import Variable
from .discretize import discretize
import pandas as pd
import numpy as np
from scipy.sparse import hstack



class Scorecard:
    @classmethod
    def discretize(
        cls,
        df: pd.DataFrame,
        perf: Performance,
        missing: float = np.nan,
        exceptions: Optional[Union[Dict[str, List[Any]], List[Any]]] = None,
        **kwargs
    ):
        variables = discretize(df, perf, missing, exceptions, **kwargs)
        return cls(variables, perf)

    def __init__(self, variables: Dict[str, Variable], perf: Performance):
        self.perf = perf
        self.variables = variables

    def summary(self):
        # create summary dataframe for each variable
        res = [v.summary() for v in self.variables.values()]
        return pd.DataFrame.from_records(res)
    
    def to_sparse(self, df: Optional[pd.DataFrame] = None, step: List[Optional[int]] = [1]):
        res = []
        for k, v in self.variables.items():
            x = df[k] if df is not None else None
            if v.step in step:
                res.append(v.to_sparse(x))
        return hstack(res)


    def __getitem__(self, key):
        return self.variables[key]