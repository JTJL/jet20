"""

"""
from numbers import Real
from typing import Union
import numpy as np
from jet20.frontend.const import *
from functools import wraps


# def constraint_check(binary_op):
#     """
#     """
#
#     @wraps(binary_op)
#     def check(self, other):
#         """
#         """
#         if isinstance(self, Expression) and self.is_constraint:
#             raise NotImplementedError("unsupported operate on constraint")
#         if isinstance(other, Expression) and other.is_constraint:
#             raise NotImplementedError("unsupported operate on constraint")
#
#         return binary_op(self, other)
#
#     return check


# TODO: 增加一个判断是否为方阵的装饰器?若不是则raise，好像装饰器做不了这个事？我需要的是golang的defer

# TODO: 增加一个通过expand self, other，让shape相同的装饰器？self不会有问题吗？

class Expression(object):
    # TODO: 补齐这里的注释
    """express a mathematical expression or formula in a square matrix form
    As shown in the following matrix, the
    [[0. 0.  0.  0. ]
     [0. 0.  0.  0.5]
     [0. 0.  0.  0.5]
     [0. 0.5 0.5 0. ]]

    The Expression overloads +, -, *, /, -(neg), **, ==, <, <=, >, >= operators
    for a convenient way to creating expressions.
    """

    def __init__(self, mat: Union[np.ndarray, None], var_index: int = 0, para_val: Real = 1.):  # op: str = '',
        """通过matrix构造
            1. 只支持n元2次齐次或n元2次非齐次方程的表达；
            2. 可表示目标函数和约束函数；
            3. 用一个二维矩阵维护n元2次方程的信息；
            4. 用额外一个constraint字段维护约束关系；
            5. matrix从row, col索引0开始表示variable的序号；
            6. matrix总会带const占位row, column，末行和末列是const占位列；
            7. 一定是方阵，次对角线相加是表达式中对应i,j变量相乘的系数

            通过参数构造(仅支持构造单一variable的1次幂形式)
        """
        if mat is not None:
            if len(mat.shape) is not 2:
                raise TypeError("mat must be a 2 dimension matrix")
            self._core_mat = mat  # also called core matrix
        else:  # construct mat
            # len=index+1, const占位=len+1, 所以此处+2
            # TODO: 全0方阵的构造是否有更简单的方法
            self._core_mat = np.zeros((var_index + 2, var_index + 2))
            # mat[i,j] = mat[j,i] = para_val/2
            # i = index + 1
            # j = index
            self._core_mat[var_index + 1, var_index] = self._core_mat[var_index, var_index + 1] = para_val / 2

        # self._op = op  # TODO: overloads ==, <, <=, >, >= 的时候实现这里

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
        if not np.array_equal(self._quadratic_matrix, np.zeros(self._quadratic_matrix.shape)):
            return 2
        elif not np.array_equal(self._linear_partial_sum, np.zeros(self._linear_partial_sum.shape)):
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
        if len(vec) >= n:
            raise ValueError("expand to a shorter vector than itself is not allowed")

        base = np.zeros(n)
        base[:len(vec)] = vec
        return base

    def _expand_linear_vector(self, n: int) -> np.ndarray:
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

    # @constraint_check
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

    # @constraint_check
    def __radd__(self, other: 'Expression') -> 'Expression':
        return self + other

    # @constraint_check
    def __sub__(self, other: 'Expression') -> 'Expression':
        if not isinstance(other, (Expression, int, float)):
            raise NotImplementedError(
                f"unsupported operand type(s) for -: {type(self).__name__} and {type(other).__name__}")
        return self + -other

    # @constraint_check
    def __rsub__(self, other: 'Expression'):
        return -self + other

    # @_cast_other
    # @constraint_check
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
                _self = self._expand_linear_vector(len(_other))
            else:
                _other = other._expand_linear_vector(len(_self))

            base = np.outer(_self, _other)
            return Expression((base.transpose() + base) / 2)
        else:
            raise NotImplementedError(
                f"unsupported operand type(s) for *: {type(self).__name__} and {type(other).__name__}")

    # @constraint_check
    def __rmul__(self, other) -> 'Expression':
        return self * other

    # all kinds of div situations are unsupported
    # def __div__(self, other) -> 'Expression':
    #     pass  # 一次幂除以一次幂？
    #
    # def __rdiv__(self, other) -> 'Expression':
    #     return other / self

    # @constraint_check
    def __pow__(self, power: int, modulo=None) -> 'Expression':
        if self.highest_order > 1:
            raise NotImplementedError("the result will exceed quadratic, which is unsupported now")

        _self = self.linear_complete_vector
        base = np.outer(_self, _self)
        return Expression((base.transpose() + base) / 2)

    # @constraint_check
    def __neg__(self) -> 'Expression':
        return self * -1

    # @constraint_check
    def __eq__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_EQUAL)

    # @constraint_check
    def __ge__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_GE)

    # @constraint_check
    def __gt__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_GT)

    # @constraint_check
    def __le__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_LE)

    # @constraint_check
    def __lt__(self, other) -> 'Constraint':
        return Constraint(self, other, OP_LE)

    def __str__(self):
        return f"{self.core_mat.__str__()} (mat)"


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
        return -expr, OP_PAIRS[self.op] if self.op in OP_PAIRS else expr, self.op
