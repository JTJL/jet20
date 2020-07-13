"""

"""
from numbers import Real
from typing import Union
import numpy as np
from jet20.frontend.const import *


class Expression(object):
    """express a mathematical expression or formula in a square matrix form
    As shown in the following matrix, the
    [[0. 0.  0.  0. ]
     [0. 0.  0.  0.5]
     [0. 0.  0.  0.5]
     [0. 0.5 0.5 0. ]]

    The Expression overloads +, -, *, /, -(neg), **, ==, <, <=, >, >= operators
    for a convenient way to creating expressions.
    """

    def __init__(self, mat: Union[np.ndarray, None]):
        if mat.ndim != 2:
            raise TypeError("mat must be a 2 dimension matrix")
        self._core_mat = mat


    @property
    def core_mat(self) -> np.ndarray:
        return self._core_mat

    # @op.setter
    # def op(self, op: str):
    #     self._op = op

    # @property
    # def is_constraint(self) -> bool:
    #     return self.op != ''

    @property
    def shape(self) -> tuple:
        """

        Returns: The shape of the core matrix

        """
        return self.core_mat.shape

    @property
    def dim(self) -> int:
        """

        Returns: The dim of the core matrix(core matrix must be a square matrix)

        """
        return self.shape[0]

    @property
    def total_variables(self) -> int:
        """

        Returns: The total numbers of variables contained in Expression

        """
        return self.dim - 1

    @property
    def highest_order(self) -> int:
        """

        Returns: The highest power of Expression

        """
        # if not np.array_equal(self._quadratic_matrix, np.zeros(self._quadratic_matrix.shape)):
        if not (self._quadratic_matrix == 0.0).all():
            return 2
        # elif not np.array_equal(self._linear_partial_sum, np.zeros(self._linear_partial_sum.shape)):
        elif not (self._linear_partial_sum == 0.0).all():
            return 1
        else:
            return 0

    @property
    def _quadratic_matrix(self) -> np.ndarray:
        """

        Returns:
        A partial matrix from the core matrix for the parameters of quadratic expression
        As shown in the following matrix, the returned matrix is in the brackets annotation
        [[0.  0.  0.]  0.      0.  0.  0.
         [0.  0.  0.]  0.5  => 0.  0.  0.
         [0.  0.  0.]] 0.5     0.  0.  0.
          0. 0.5 0.5   0.
        """
        return self.core_mat[:self.shape[0] - 1, :self.shape[1] - 1]

    @property
    def _linear_row_partial(self) -> np.ndarray:
        """

        Returns:
        A vector extracted form the core matrix for row partial of linear expr's parameters.
        The value in this vector means the (correct parameter)/2.
        As shown in the following matrix, the returned vector is in the brackets annotation
         0.  0.  0.  0.
         0.  0.  0.  0.5  => [0., 0.5, 0.5]
         0.  0.  0.  0.5
        [0. 0.5 0.5] 0.
        """
        return self.core_mat[self.shape[0] - 1, :self.shape[0] - 1]

    @property
    def _linear_col_partial(self) -> np.ndarray:
        """

        Returns:
        A vector extracted form the core matrix for column partial of linear expr's parameters.
        The value in this vector means the (correct parameter)/2.
        As shown in the following matrix, the returned vector is in the brackets annotation
        0. 0.  0.  [0. ]
        0. 0.  0.  [0.5]  => [0., 0.5, 0.5]
        0. 0.  0.  [0.5]
        0. 0.5 0.5  0.
        """
        return self.core_mat[:self.shape[0] - 1, self.shape[0] - 1]

    @property
    def _linear_partial_sum(self) -> np.ndarray:
        """

        Returns:
        A vector for added row and column partial of the the linear variable parameters.
        As shown in the following matrix, the returned vector is the sum of the two brackets annotated vectors.
         0. 0.  0.   [0.]
         0. 0.  0.   [0.5]  => [0., 1., 1.]
         0. 0.  0.   [0.5]
        [0. 0.5 0.5]  0.

        """
        return self._linear_row_partial + self._linear_col_partial

    @property
    def const(self) -> Real:
        """

        Returns:
        The value of const in the expression.
        As shown in the following matrix, the returned value is the brackets annotation
        0. 0.  0.   0.
        0. 0.  0.   0.5  => 0.
        0. 0.  0.   0.5
        0. 0.5 0.5 [0.]
        """
        return self.core_mat[self.shape[0] - 1, self.shape[1] - 1]

    @property
    def linear_complete_vector(self) -> np.ndarray:
        """'complete' means the returned linear vector contains const value
        Returns:
        A vector for complete parameters of the linear variable and const value
        As shown in the following matrix, the returned vector is the brackets annotation
         0. 0.  0.   [0.]
         0. 0.  0.   [0.5]  => [0., 1., 1., 0.]
         0. 0.  0.   [0.5]
        [0. 0.5 0.5] [0.]

        """
        return np.append(self._linear_partial_sum, self.const)

    @staticmethod
    def _expand_vector(vec: np.ndarray, n: int) -> np.ndarray:
        """Expand a vector to (n,1) shape with zeros
        Args:
            n: expand to n*1 dim(n must larger than the length of vec).
            vec: Vector will be expand.

        Returns:
            A new np.ndarray expanded base on the input vec
        """
        if len(vec) > n:
            raise ValueError("expand to a shorter vector than itself is not allowed")
        if len(vec) == n:
            return vec

        base = np.zeros(n)
        base[:len(vec)] = vec
        return base

    def expand_linear_vector(self, n: int) -> np.ndarray:
        """

        Args:
            n: expand to n*1 dim.

        Returns:
            A new linear parameters vector expanded with zeros(compounded with const)
        """
        return np.append(self._expand_vector(self._linear_partial_sum, n - 1), self.const)

    def expand(self, n: int) -> 'Expression':
        """Get a new Expression which's core matrix has been expanded to n*n dim. In Mathematically speaking,
        expand means add variables into expression.
        Args:
            n: expand to n*n dim

        Returns:
            A new core matrix has been expanded to n*n dim
        """
        if n < self.dim:
            raise ValueError("expand dim less than the current is not allowed")
        if n == self.dim:
            return self

        base = np.zeros((n, n))
        _q_mat = self._quadratic_matrix
        base[:_q_mat.shape[0], :_q_mat.shape[1]] = _q_mat  # cover quadratic paras back

        # cover linear row partial back
        base[base.shape[0] - 1, :self._linear_row_partial.shape[0]] = self._linear_row_partial

        # cover linear column partial back
        base[:self._linear_col_partial.shape[0], base.shape[1] - 1] = self._linear_col_partial
        base[base.shape[0] - 1, base.shape[1] - 1] = self.const  # cover const back
        return Expression(base)

    def equal(self, other: 'Expression') -> bool:
        if isinstance(self.core_mat, np.ndarray) and isinstance(other.core_mat, np.ndarray):
            return np.array_equal(self.core_mat, other.core_mat)
        return False

    def __add__(self, other: Union[Real, 'Expression']) -> 'Expression':
        if isinstance(other, (int, float)):  # Expression + number
            base = self.core_mat.copy()
            base[base.shape[0] - 1, base.shape[1] - 1] = other
            return Expression(base)
        elif isinstance(other, Expression):  # Expression + Expression
            # align
            _self = self.core_mat
            _other = other.core_mat
            if self.dim < other.dim:
                _self = self.expand(other.dim).core_mat
            else:
                _other = other.expand(self.dim).core_mat

            return Expression(_self + _other)
        else:
            raise NotImplementedError(
                f"unsupported operand type(s) for +: {type(self).__name__} and {type(other).__name__}")

    def __radd__(self, other: 'Expression') -> 'Expression':
        return self + other

    def __sub__(self, other: 'Expression') -> 'Expression':
        if not isinstance(other, (Expression, int, float)):
            raise NotImplementedError(
                f"unsupported operand type(s) for -: {type(self).__name__} and {type(other).__name__}")
        return self + -other

    def __rsub__(self, other: 'Expression'):
        return -self + other

    def __mul__(self, other: Union[Real, 'Expression']) -> 'Expression':
        if isinstance(other, (int, float)):  # number * Expression
            return Expression(self.core_mat * other)
        elif isinstance(other, Expression):  # Expression * Expression
            if self.highest_order > 1 or other.highest_order > 1:
                raise NotImplementedError(
                    f"the result will exceed quadratic, which is unsupported now")

            _self = self.linear_complete_vector
            _other = other.linear_complete_vector
            # align two vectors
            if len(_self) < len(_other):
                _self = self.expand_linear_vector(len(_other))
            else:
                _other = other.expand_linear_vector(len(_self))

            base = np.outer(_self, _other)
            return Expression((base.transpose() + base) / 2)
        else:
            raise NotImplementedError(
                f"unsupported operand type(s) for *: {type(self).__name__} and {type(other).__name__}")

    def __rmul__(self, other) -> 'Expression':
        return self * other

    # all kinds of div situations are unsupported
    # def __div__(self, other) -> 'Expression':
    #     pass  # 一次幂除以一次幂？
    #
    # def __rdiv__(self, other) -> 'Expression':
    #     return other / self

    def __pow__(self, power: int, modulo=None) -> 'Expression':
        if self.highest_order > 1:
            raise NotImplementedError("the result will exceed quadratic, which is unsupported now")

        _self = self.linear_complete_vector
        base = np.outer(_self, _self)
        return Expression((base.transpose() + base) / 2)

    def __neg__(self) -> 'Expression':
        return self * -1

    def __eq__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_EQUAL)

    def __ge__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_GE)

    def __gt__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_GT)

    def __le__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_LE)

    def __lt__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_LE)

    def __str__(self):
        return f"{self.core_mat.__str__()} (mat)"

    __repr__ = __str__


class Constraint(object):
    def __init__(self, lh: Union[Expression, float], rh: Union[Expression, float], op: OP):
        self._lh: Union[Expression, float] = lh
        self._rh: Union[Expression, float] = rh
        self._op: OP = op

    @property
    def lh(self) -> Expression:
        return self._lh

    @property
    def rh(self) -> Expression:
        return self._rh

    @property
    def op(self) -> str:
        return self._op

    def with_op(self, op: str) -> 'Constraint':
        self._op = op
        return self

    def canonicalize(self) -> (Expression, OP):
        expr = self._lh - self._rh
        # lh > rh => -(lh-rh) < 0
        # lh < rh => lh-rh < 0
        return (-expr, OP_PAIRS[self.op]) if self.op in OP_PAIRS else (expr, self.op)






