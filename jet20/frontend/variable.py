from jet20.frontend.expression import Expression, Constraint
from jet20.frontend.const import *
from typing import List, Union, Optional
from numbers import Real
import numpy as np


class Variable(Expression):
    """

    """

    def __init__(self, index: int, name: str, lb: Optional[float]=None,ub: Optional[float]=None,coef: Real = 1.):
        # len=index+1, const占位=len+1, 所以此处+2
        # TODO: 全0方阵的构造是否有更简单的方法
        _core_mat = np.zeros((index + 2, index + 2))
        # mat[i,j] = mat[j,i] = para_val/2
        # i = index + 1
        # j = index
        _core_mat[index + 1, index] = _core_mat[index, index + 1] = coef / 2.0
        Expression.__init__(self, _core_mat)

        self._index: int = index
        self._name: str = name
        self._lb: Optional[float] = lb
        self._ub: Optional[float] = ub
        # self._constraints: List[Constraint] = []
        # TODO: upper/lower bound, constraint_type([0,1], integer, float...)

    @property
    def lb(self) -> Optional[float]:
        return self._lb

    @property
    def ub(self) -> Optional[float]:
        return self._ub

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == "matmul" and method == "__call__":
            return inputs[1].__rmatmul__(inputs[0])
        else:
            raise NotImplementedError("")

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self):
        for x in self._arr:
            yield x

    def __getitem__(self,k):
        return self._arr[k]


    def append(self, element: Union[Variable, Expression, Constraint]):
        if not (isinstance(element, Variable) or isinstance(element, Expression)):
            raise TypeError("element must be a Variable or a Expression")
        self._arr += [element]


    def __rmatmul__(self, other: Union[np.ndarray, 'Array']) -> 'Array':
        if isinstance(other, np.ndarray):  # parameters @ variables
            if other.shape[-1] != len(self):  # TODO: shape的合法性检查
                raise ValueError("mismatch length of parameter array and variable array")
            if other.ndim == 1:
                return other @ self._arr
            elif other.ndim == 2:
                return Array( [r @ self._arr for r in other] )
            else:
                raise NotImplementedError("")

        elif isinstance(other, Array):  # variables @ variables
            return Array([ a * b for a,b in zip(other._arr,self._arr)])

        else:
            raise NotImplementedError("")

    def __lt__(self, other) -> 'Array':
        if isinstance(other, (float,int)):
            return Array([expr < float(other) for expr in self.array])  # constraint list

        if isinstance(other, (list,np.ndarray)):
            if len(other) != len(self):
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr < float(other[i]) for i, expr in enumerate(self.array)])

    def __le__(self, other) -> 'Array':
        if isinstance(other, (float,int)):
            return Array([expr <= float(other) for expr in self.array])  # constraint list

        if isinstance(other, (list,np.ndarray)):
            if len(other) != len(self):
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr <= float(other[i]) for i, expr in enumerate(self.array)])

    def __gt__(self, other) -> 'Array':
        if isinstance(other, (float,int)):
            return Array([expr > float(other) for expr in self.array])  # constraint list

        if isinstance(other, (list,np.ndarray)):
            if len(other) != len(self):
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr > float(other[i]) for i, expr in enumerate(self.array)])

    def __ge__(self, other) -> 'Array':
        if isinstance(other, (float,int)):
            return Array([expr >= float(other) for expr in self.array])  # constraint list

        if isinstance(other, (list,np.ndarray)):
            if len(other) != len(self):
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr >= float(other[i]) for i, expr in enumerate(self.array)])

    def __eq__(self, other) -> 'Array':
        if isinstance(other, (float,int)):
            return Array([expr == float(other) for expr in self.array])  # constraint list

        if isinstance(other, (list,np.ndarray)):
            if len(other) != len(self):
                raise ValueError("mismatch length of parameter array and variable array")
            return Array([expr == float(other[i]) for i, expr in enumerate(self.array)])


def quad(q: np.ndarray,xs: 'Array') -> 'Expression':
    if q.ndim != 2 or q.shape[-1] != len(xs):
        raise NotImplementedError("")
    
    low,upper = xs[0].index,xs[-1].index
    for i,x in enumerate(xs):
        if not isinstance(x,Variable):
            raise ValueError("only array of Variable supported")
        if x.index != low+i:
            raise ValueError("Variable index must be continuous")

    size = upper + 2
    coef_mat = np.zeros((size,size))
    coef_mat[low:upper+1,low:upper+1] = q
    return Expression(coef_mat)

        
    