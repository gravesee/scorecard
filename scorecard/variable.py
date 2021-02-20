# a variable manages a series of transforms, fit objects, history,
# and allows for step 1, 2, NA, constraints, neutralize etc...

from .transform import Transform
import copy


def undoable(fun):
    def inner(self, *args, **kwargs):
        self.history.append(copy.deepcopy(self.transform))
        fun(self, *args, **kwargs)

    return inner


class Variable:
    def __init__(self, transform: Transform):
        self.transform = transform
        self.fit_object = None
        self.constraints = None
        self.neutralize = {}
        
        self._step = None
        self._mono = 0
        
        self.history = []
    
    @property
    def step(self):
        return self._step
    
    @step.setter
    def step(self, value):
        if value not in {1, 2, None}:
            raise Exception("Step must be in {1, 2, None}")
        self._step = value
    
    @property
    def mono(self):
        return self._mono
    
    @mono.setter
    def mono(self, value):
        if value not in {-1, 0, 1}:
            raise Exception("Monotonicity must be in {-1, 0, 1}")
        self._mono = value

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
        if len(self.history) > 0:
            self.transform = self.history.pop()
