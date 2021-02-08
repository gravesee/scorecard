from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Tuple

import numpy as np
import copy
import pandas as pd
import scipy.sparse as sp
import pickle


class Transform(ABC):
    def __init__(self, exceptions: List[Any], missing: Any):
        self.exceptions = exceptions
        self.missing = missing
        self._init = pickle.dumps(self)

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
        return pd.Series(cat)  # type: ignore

    def reset(self):
        init = pickle.loads(self._init)
        self.__dict__ = init.__dict__


class ContinuousTransform(Transform):
    def __init__(
        self, breaks: List[float], exceptions: List[float], missing: float
    ) -> None:
        self.breaks = breaks
        super().__init__(exceptions, missing)

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
        """collapse bins together

        Args:
            indices (List[int]): index positiong of bins to collapse

        The bins are indexed the same as the labels. Only lowest and highest indices
        are used to collapse the entire range in-beteween.
        """
        idx = sorted(indices)
        rng = list(range(idx[0], idx[-1]))
        breaks = []
        for i, x in enumerate(self.breaks[1:]):
            if i not in rng:
                breaks.append(x)

        self.breaks = breaks

    def expand(self, index: int, value: float):
        """expand bin into two bins


        """
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
        self.levels = [[x] for x in levels]
        super().__init__(exceptions, missing)

    @property
    def _labels(self):
        return list(map(str, self.levels))

    def collapse(self, indices: List[int]):
        """collapse bins together

        Args:
            indices (List[int]): index positiong of bins to collapse

        The bins are indexed the same as the labels. All selected bins are combined
        into one new bin. The old bins are removed. The constituents are maintained, though,
        so the original bins can be retrieved with an expand operation.
        """
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
            out[np.isin(x, els)] = i  # type: ignore

        return out, i  # type: ignore
