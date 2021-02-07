from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Transform(ABC):
    @abstractproperty
    def labels(self) -> List[str]:
        pass

    @abstractmethod
    def to_integer(self, x: pd.Series) -> pd.Series:
        pass
    
    @abstractmethod
    def collapse(self, indices: List[int]):
        pass
    
    @abstractmethod
    def expand(self, index: int, **kwargs):
        pass

    def to_sparse(self, x: pd.Series) -> sp.csc_matrix:
        M, N = len(x), len(self.labels)
        col_ind = self.to_integer(x)
        row_ind = np.arange(len(x))
        data = np.ones(len(x))
        return sp.csc_matrix((data, (row_ind, col_ind)), (M, N))

    def to_categorical(self, x: pd.Series) -> pd.Series:
        ix = self.to_integer(x)
        return pd.Series(self.labels[ix], dtype="category")


class ContinuousTransform(Transform):
    def __init__(
        self, breaks: List[float], exceptions: List[float], missing: float
    ) -> None:
        breaks.insert(0, -np.inf)
        breaks.append(np.inf)
        self.breaks = np.unique(breaks)
        self.exceptions = exceptions
        self.missing = missing

    @property
    def labels(self):
        labels = []
        for start, stop in zip(self.breaks, self.breaks[1:]):
            labels.append(str((start, stop)))
        return np.array(labels)

    def to_integer(self, x: pd.Series):
        out = np.full_like(x, fill_value=np.nan, dtype=int)
        for i, (start, stop) in enumerate(zip(self.breaks, self.breaks[1:])):
            f = (x >= start) & (x <= stop)
            out[f] = i
        return out

class CategoricalTransform(Transform):
    def __init__(self, levels: List[Any]):
        self.levels = [[x] for x in levels]

    @property
    def labels(self):
        labels = list(map(str, self.levels))
        return np.array(labels)

    def collapse(self, indices: List[int]):
        for ix in indices[1:]:
            self.levels[indices[0]] += self.levels[ix]

        self.levels = [l for i, l in enumerate(self.levels) if i not in indices[1:]]
        self.levels = sorted(self.levels)

    def expand(self, index: int):
        levels = self.levels.pop(index)
        self.levels += [[l] for l in levels]
        self.levels = sorted(self.levels)

    def to_integer(self, x: pd.Series):
        out = np.full_like(x, fill_value=np.nan, dtype=int)
        for i, els in enumerate(self.levels):
            out[np.isin(x, els)] = i
        return out


v = CategoricalTransform(list("abcde"))

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_integer(pd.Series(x))

v.to_categorical(pd.Series(x))

v.to_sparse(pd.Series(x))
