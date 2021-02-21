# a variable manages a series of transforms, history,
# and allows for step 1, 2, NA, constraints, neutralize etc...
from typing import Dict, List, Set, Tuple
from .transform import ContinuousTransform, Transform
import copy
from .performance import Performance
import pandas as pd


def undoable(fun):
    def inner(self: "Variable", *args, **kwargs):
        self._history.append(copy.deepcopy(self.transform))
        fun(self, *args, **kwargs)

    return inner


class Variable:
    def __init__(
        self,
        x: pd.Series,
        perf: Performance,
        transform: Transform,
        neutralize_missing: bool = True,
    ):
        self.x = x
        self.perf = perf
        self.transform = transform

        self._constraints: Dict[str, Tuple[str, str]] = {}

        self._step = None
        self._history = []

        if neutralize_missing:
            self._constraints["Missing"] = ("Missing", "neu")

    @property
    def name(self):
        return self.x.name

    @property
    def type(self):
        if isinstance(self.transform, ContinuousTransform):
            return "num"
        else:
            return "cat"

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value not in {1, 2, None}:
            raise Exception("`step` must be one of {1, 2, None}")
        self._step = value

    def get_constraints(self, value) -> List[Dict]:
        # translate labels to integer positions if the labels still exist
        labels = self.transform.labels
        res = []
        for base, (target, op) in self._constraints.items():
            if base in labels and target in labels:
                type_ = "eq" if op == "=" else "ineq"
                i, j = labels.index(target), labels.index(base)
                indices = (j, i) if op == "<" else (i, j)
                res.append({"type": type_, "indices": indices, "len": len(labels)})

        return res

    def get_constraint_repr(self):
        """return series representation of constraints for display purposes"""
        labels = self.transform.labels
        res = pd.Series(index=labels, dtype=str)
        for label in labels:
            if self._constraints.get(label, None) is not None:
                target, op = self._constraints[label]
                if op == "neu":
                    res[label] = f"{op}"
                else:
                    res[label] = f"{op} {labels.index(target)}"
        return res.fillna("").rename("Constr")

    def set_constraint(self, base: int, target: int, op: str):
        labels = self.transform.labels
        if (base < 0) or (base > len(labels) - 1):
            raise Exception(f"invalid level used in variable operation: {base}")

        if (target < 0) or (target > len(labels) - 1):
            raise Exception(f"invalid level used in variable operation: {target}")

        if op not in {"<", "=", ">", "neu"}:
            raise Exception(
                "invalid constraint operation. must be one of {<, >, =, neu}"
            )

        self._constraints[labels[base]] = (labels[target], op)

    def remove_constraint(self, base: int):
        labels = self.transform.labels
        if labels[base] in self._constraints:
            del self._constraints[labels[base]]

    def _monotonic_constraints(self, op: str):
        # self.clear_constraints()
        # labels excluding exceptions, missing
        indices = range(len(self.transform._labels))
        for i, j in zip(indices, indices[1:]):
            self.set_constraint(i, j, op)

    def increasing_constraints(self):
        self._monotonic_constraints("<")

    def decreasing_constraints(self):
        self._monotonic_constraints(">")

    def clear_constraints(self):
        self._constraints.clear()

    def neutralize(self, index: int):
        labels = self.transform.labels
        if (index < 0) or (index > len(labels) - 1):
            raise Exception(f"invalid level used in variable operation: {index}")
        self.set_constraint(index, index, "neu")

    @undoable
    def collapse(self, indices):
        self.transform.collapse(indices)

    @undoable
    def expand(self, index, **kwargs):
        self.transform.expand(index, **kwargs)

    @undoable
    def reset(self):
        self.transform.reset()

    def undo(self):
        if len(self._history) > 0:
            self.transform = self._history.pop()

    def __str__(self):
        s = self.transform.to_categorical(self.x)
        res = self.perf.summarize(s)
        con = self.get_constraint_repr()

        return str(pd.concat([res, con], axis=1).fillna(""))  # type: ignore

    def summary(self):
        name, stat = self.perf.summary_statistic(self.x)
        return {
            "Variable": self.name,
            "Step": self.step,
            "Type": self.type,
            "Len": len(self.transform),
            name: stat,
        }
