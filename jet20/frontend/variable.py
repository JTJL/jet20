from jet20.frontend.expression import Expression, Constraint
from jet20.frontend.const import *
from typing import List, Union
import numpy as np


class Variable(Expression):
    """

    """

    def __init__(self, index: int, name: str):
        Expression.__init__(self, None, var_index=index)
        self._index: int = index
        self._name: str = name
        self._constraints: List[Constraint] = []
        # TODO: upper/lower bound, constraint_type([0,1], integer, float...)

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self):
        pass

    def __str__(self):
        return f"{super().__str__()} (expr), {self.name} (name), {self.index} (index)"


class Array(object):
    def __init__(self, array: List[Union[Variable, Expression, Constraint]] = None):
        self._arr: List[Union[Variable, Expression, Constraint]] = array if array else []

    @property
    def array(self) -> List[Union[Variable, Expression, Constraint]]:
        return self._arr

    @property
    def len(self) -> int:
        return len(self._arr)

    def append(self, element: Union[Variable, Expression, Constraint]):
        if not (isinstance(element, Variable) or isinstance(element, Expression)):
            raise TypeError("element must be a Variable or a Expression")
        self._arr += [element]

    def __matmul__(self, other: Union[np.ndarray, 'Array']) -> 'Array':
        if isinstance(other, np.ndarray):  # parameters @ variables
            if other.shape[1] != self.len:  # TODO: shape的合法性检查
                raise ValueError("mismatch length of parameter array and variable array")

            new_arr = Array()
            for row in other:
                new_arr.append(sum([self.array[i] * row[i] for i in range(len(row))]))  # expr list
            return new_arr
        if isinstance(other, Array):  # variables @ variables
            pass

    def __lt__(self, other) -> 'Array':
        if isinstance(other, float):
            return Array([expr < other for expr in self.array])  # constraint list

        if isinstance(other, list):
            if len(other) != self.len:
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr < other[i] for i, expr in enumerate(self.array)])

    def __le__(self, other) -> 'Array':
        if isinstance(other, float):
            return Array([expr <= other for expr in self.array])  # constraint list

        if isinstance(other, list):
            if len(other) != self.len:
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr <= other[i] for i, expr in enumerate(self.array)])

    def __gt__(self, other) -> 'Array':
        if isinstance(other, float):
            return Array([expr > other for expr in self.array])  # constraint list

        if isinstance(other, list):
            if len(other) != self.len:
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr > other[i] for i, expr in enumerate(self.array)])

    def __ge__(self, other) -> 'Array':
        if isinstance(other, float):
            return Array([expr >= other for expr in self.array])  # constraint list

        if isinstance(other, list):
            if len(other) != self.len:
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr >= other[i] for i, expr in enumerate(self.array)])

    def __eq__(self, other) -> 'Array':
        if isinstance(other, float):
            return Array([expr == other for expr in self.array])  # constraint list

        if isinstance(other, list):
            if len(other) != self.len:
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr == other[i] for i, expr in enumerate(self.array)])
