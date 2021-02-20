# a variable manages a series of transforms, history,
# and allows for step 1, 2, NA, constraints, neutralize etc...

from typing import Dict, Set, Tuple
from .transform import Transform
import copy


def undoable(fun):
    def inner(self: Variable, *args, **kwargs):
        self._history.append(copy.deepcopy(self.transform))
        fun(self, *args, **kwargs)

    return inner


class Variable:
    def __init__(self, transform: Transform):
        self.transform = transform
        self.constraints: Dict[str, Tuple[str, str]]
        self.neutralize: Set[str] = set()

        self._step = None
        self._mono = 0

        self._history = []

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value not in {1, 2, None}:
            raise Exception("`step` must be one of {1, 2, None}")
        self._step = value

    @property
    def mono(self):
        return self._mono

    @mono.setter
    def mono(self, value):
        if value not in {-1, 0, 1}:
            raise Exception("`mono` must be one of {-1, 0, 1}")
        self._mono = value

    def get_constraints(self, value):
        pass

    def set_constraint(self, base: int, target: int, op: str):
        labels = self.transform.labels
        if (base < 0) or (base > len(labels)):
            raise Exception(f"invalid level used in variable operation: {base}")

        if (target < 0) or (target > len(labels)):
            raise Exception(f"invalid level used in variable operation: {target}")

        if op not in {"<", "=", ">"}:
            raise Exception("invalid constraint operation. must be one of {<, >, =}")

        self.constraints[labels[base]] = (labels[target], op)
    
    def remove_constraint(self, base: int):
        labels = self.transform.labels
        if labels[base] in self.constraints:
            del self.constraints[labels[base]]
    
    def clear_constraints(self):
        self.constraints = {}

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
