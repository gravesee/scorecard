from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Tuple

import numpy as np
import copy
import pandas as pd
import scipy.sparse as sp


class Transform(ABC):
    def __init__(self, exceptions: List[Any], missing: Any):
        self.exceptions = exceptions
        self.missing = missing

    @abstractproperty
    def _labels(self) -> List[str]:  # type: ignore
        pass

    @property
    def labels(self) -> List[str]:
        labels = self._labels
        for e in self.exceptions:
            labels.append(e)
        labels.append("Missing")
        return labels

    def __len__(self) -> int:
        return len(self.labels)

    def is_missing(self, x: pd.Series):
        if np.isnan(self.missing):
            return np.isnan(x)
        else:
            return x == self.missing

    def to_index(self, x: pd.Series) -> np.ndarray:
        out, i = self._to_index(x)

        for e in self.exceptions:
            i += 1
            out[x == e] = i

        out[self.is_missing(x)] = i + 1

        return out

    @abstractmethod
    def _to_index(self, x: pd.Series) -> pd.Series:
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
        super().__init__(exceptions, missing)
        self.breaks = breaks

    @property
    def breaks(self):
        return self._breaks

    @breaks.setter
    def breaks(self, breaks):
        breaks.insert(0, -np.inf)
        breaks.append(np.inf)
        self._breaks = sorted(set(breaks))

    @property
    def _labels(self):
        labels = []
        for start, stop in zip(self.breaks, self.breaks[1:]):
            labels.append(str((start, stop)))
        return labels

    def collapse(self, indices: List[int]):
        idx = sorted(indices)
        rng = list(range(idx[0], idx[-1]))
        breaks = []
        for i, x in enumerate(self.breaks[1:]):
            if i not in rng:
                breaks.append(x)

        self.breaks = breaks

    def expand(self, index: int, value: float):
        breaks = copy.copy(self.breaks)
        breaks.insert(index, value)
        self.breaks = breaks

    def _to_index(self, x: pd.Series) -> Tuple[np.ndarray, int]:
        out: np.ndarray = np.full_like(x, fill_value=np.nan, dtype=int)  # type: ignore
        for i, (start, stop) in enumerate(zip(self.breaks, self.breaks[1:])):
            f = (x >= start) & (x <= stop)
            out[f] = i

        return out, i


class CategoricalTransform(Transform):
    def __init__(self, levels: List[Any], exceptions: List[Any], missing: float):
        super().__init__(exceptions, missing)
        self.levels = [[x] for x in levels]

    @property
    def _labels(self):
        return list(map(str, self.levels))

    def collapse(self, indices: List[int]):
        for ix in indices[1:]:
            self.levels[indices[0]] += self.levels[ix]

        self.levels = [l for i, l in enumerate(self.levels) if i not in indices[1:]]
        self.levels = sorted(self.levels)

    def expand(self, index: int):
        levels = self.levels.pop(index)
        self.levels += [[l] for l in levels]
        self.levels = sorted(self.levels)

    def _to_index(self, x: pd.Series) -> Tuple[np.ndarray, int]:
        i, j = 0, 0
        out = np.full_like(x, fill_value=np.nan, dtype=int)
        for i, els in enumerate(self.levels):
            out[np.isin(x, els)] = i

        return out, i


v = ContinuousTransform([-3, -2, -1, 0, 1, 2, 3], [-998], np.nan)

v.labels
v.expand(0, -5)
v.collapse([0, 1])

x = np.random.randn(10000)

pd.Series(v.to_index(x)).value_counts()

v.labels
v.collapse([0, 2])

v.labels

v.to_categorical(x).value_counts(sort=False)


v = CategoricalTransform(list("abcde"), ["Z"], -998)

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_index(pd.Series(x))
v.to_categorical(pd.Series(x))
v.to_sparse(pd.Series(x))