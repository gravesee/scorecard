# TODO: copy important bits form pycard

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple
import pandas as pd
import numpy as np
import hashlib


def series_cache(fun):
    def inner(self, s, cache=True):
        key = None
        if cache:
            if hasattr(s, "cat"):
                key = hashlib.md5(s.cat.codes.values).hexdigest()
            else:
                key = hashlib.md5(s.values).hexdigest()
            if self._cache.get(key, None) is not None:
                return self._cache[key]

        res = fun(self, s)
        if cache:
            self._cache[key] = res
        return res

    return inner


class Performance(ABC):
    def __init__(self, y, w):
        self.y = pd.Series(y)
        if w is None:
            w = np.ones_like(y)
        self.w = pd.Series(w)
        self._cache = {}

    @abstractmethod
    def summarize(self, s: pd.Series, cache: bool = True):
        pass

    @abstractproperty
    def summary_statistic(self):
        pass

    def __iter__(self):
        return iter((self.y, self.w))


class BinaryPerformance(Performance):
    def __init__(self, y, w=None) -> None:
        super().__init__(y, w)

    @series_cache
    def summarize(self, x: pd.Series, cache: bool = True):
        cnts = (
            self.w.groupby([x, self.y])
            .count()
            .unstack()
            .rename(columns={0: "# 0s", 1: "# 1s"})
        )
        tots = cnts.sum(axis=1).rename("N")
        pcts = (cnts / cnts.sum()).rename(columns={"# 0s": "% 0s", "# 1s": "% 1s"})
        rate = (cnts["# 1s"] / tots).rename("1s Rate")
        woe = pd.Series(np.log(pcts["% 1s"] / pcts["% 0s"]), name="WoE")  # type: ignore
        iv = (woe * (pcts["% 1s"] - pcts["% 0s"])).rename("IV")

        res = pd.concat([tots, cnts, rate, pcts, woe, iv], axis=1)
        res.index = res.index.astype(str)
        res.loc["Total"] = res.sum(axis=0)

        # fix the total row
        tot_rate = res.loc["Total", "# 1s"] / res.loc["Total", "N"]
        res.loc["Total", ["1s Rate", "WoE", "IV"]] = [tot_rate, 0, iv.sum()]  # type: ignore

        return res.fillna(0)

    def summary_statistic(self, s) -> Tuple[str, float]:
        summary = self.summarize(s)
        return "IV", summary.loc["Total", "IV"]


class ContinuousPerformance(Performance):
    pass
