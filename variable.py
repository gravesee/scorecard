from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List
import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Transform(ABC):

    # is this necessary?
    exceptions: List[Any]
    missing: Any

    @abstractproperty
    def labels(self) -> List[str]:
        
        pass

    @abstractmethod
    def to_index(self, x: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def collapse(self, indices: List[int]):
        pass

    @abstractmethod
    def expand(self, index: int, **kwargs):
        pass

    def to_sparse(self, x: pd.Series) -> sp.csc_matrix:
        M, N = len(x), len(self.labels)
        col_ind = self.to_index(x)
        row_ind = np.arange(len(x))
        data = np.ones(len(x))
        return sp.csc_matrix((data, (row_ind, col_ind)), (M, N))

    def to_categorical(self, x: pd.Series) -> pd.Series:
        ix = self.to_index(x)
        labels = np.array(self.labels)
        cat = pd.Categorical(labels[ix], categories=labels, ordered=True)
        return pd.Series(cat)


class ContinuousTransform(Transform):
    def __init__(
        self, breaks: List[float], exceptions: List[float], missing: float
    ) -> None:
        self.breaks = breaks
        self.exceptions = exceptions
        self.missing = missing

    @property
    def breaks(self):
        return self._breaks

    @breaks.setter
    def breaks(self, breaks):
        breaks.insert(0, -np.inf)
        breaks.append(np.inf)
        self._breaks = sorted(set(breaks))

    @property
    def labels(self):
        labels = []
        for start, stop in zip(self.breaks, self.breaks[1:]):
            labels.append(str((start, stop)))
        for e in self.exceptions:
            labels.append(e)
        labels.append("Missing")
        return labels

    def collapse(self, indices: List[int]):
        idx = sorted(indices)
        rng = list(range(idx[0], idx[-1]))
        breaks = []
        for i, x in enumerate(self.breaks[1:]):
            if i not in rng:
                breaks.append(x)

        self.breaks = breaks

    def expand(self, index: int):
        pass

    def to_index(self, x: pd.Series):
        i, j = 0, 0
        out = np.full_like(x, fill_value=np.nan, dtype=int)
        for i, (start, stop) in enumerate(zip(self.breaks, self.breaks[1:])):
            f = (x >= start) & (x <= stop)
            out[f] = i

        # add exceptions
        for j, e in enumerate(self.exceptions):
            out[x == e] = i + j

        # add missing
        if np.isnan(self.missing):
            out[np.isnan(x)] = i + j + 1
        else:
            out[x == self.missing] = i + j + 1

        return out


class CategoricalTransform(Transform):
    def __init__(self, levels: List[Any], exceptions: List[Any], missing: float):
        self.levels = [[x] for x in levels]
        self.exceptions = exceptions
        self.missing = missing

    @property
    def labels(self):
        labels = list(map(str, self.levels))
        # return super().labels(labels)
        for e in self.exceptions:
            labels.append(e)
        labels.append("Missing")
        return labels

    def collapse(self, indices: List[int]):
        for ix in indices[1:]:
            self.levels[indices[0]] += self.levels[ix]

        self.levels = [l for i, l in enumerate(self.levels) if i not in indices[1:]]
        self.levels = sorted(self.levels)

    def expand(self, index: int):
        levels = self.levels.pop(index)
        self.levels += [[l] for l in levels]
        self.levels = sorted(self.levels)

    def to_index(self, x: pd.Series):
        i, j = 0, 0
        out = np.full_like(x, fill_value=np.nan, dtype=int)
        for i, els in enumerate(self.levels):
            out[np.isin(x, els)] = i

        # add exceptions
        for j, e in enumerate(self.exceptions):
            out[x == e] = i + j

        # add missing
        if np.isnan(self.missing):
            out[np.isnan(x)] = i + j + 1
        else:
            out[x == self.missing] = i + j + 1

        return out


v = ContinuousTransform([-3, -2, -1, 0, 1, 2, 3], [-998], np.nan)
x = np.random.randn(10000)

pd.Series(v.to_index(x)).value_counts()

v.labels
v.collapse([0, 2])

v.labels

v.to_categorical(x).value_counts(sort=False)


v = CategoricalTransform(list("abcde"), [], -998)

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_index(pd.Series(x))

v.to_categorical(pd.Series(x))

v.to_sparse(pd.Series(x))