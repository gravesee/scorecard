# when a model is fit, need to record some information about it

from typing import Dict
from .variable import Variable
import copy
import uuid

class Model:
    def __init__(self, obj, variables: Dict[str, Variable], name: str = ""):
        self.name = name
        self.obj = copy.deepcopy(obj)
        self.variables = copy.deepcopy(variables)
