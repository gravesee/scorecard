# TODO: copy important bits form pycard

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Performance(ABC):
    @abstractmethod
    def summarize(self, s: pd.Series):
        pass


class BinaryPerformance(Performance):
    def __init__(self, y, w) -> None:
        self.y = pd.Series(y)
        self.w = pd.Series(w)

    def summarize(self, s):
        cnts = (
            self.w.groupby([s, self.y])
            .count()
            .unstack()
            .rename(columns={0: "# 0s", 1: "# 1s"})
        )
        tots = cnts.sum(axis=1).rename("N")
        pcts = (cnts / cnts.sum()).rename(columns={0: "% 0s", 1: "% 1s"})
        woe = pd.Series(np.log(pcts["# 1s"] / pcts["# 0s"]), name="WoE")
        
        res = pd.concat([tots, cnts, pcts, woe], axis=1)
        res.index = res.index.astype(str)
        res.loc["Total"] = res.sum(axis=0)

        return res


class ContinuousPerformance(Performance):
    pass