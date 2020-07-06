from jet20.frontend.expression import Expression, Constraint
from jet20.frontend.variable import Variable, Array
from jet20.frontend.const import *
from jet20.frontend.backends import jet20_default_backend_func
from typing import List, Union, Callable
from functools import wraps
import numpy as np
import re


# TODO:提供with处理exception

# def assert_not_constraint(add_expr):
#     """
#     """
#
#     @wraps(add_expr)
#     def check(self, expr: Expression):
#         """
#         """
#         if isinstance(expr, Expression) and expr.is_constraint:
#             raise NotImplementedError("unsupported put a constraint as object")
#
#         return add_expr(self, expr)
#
#     return check


def assert_power(add_expr):
    """
    """

    @wraps(add_expr)
    def check(self, constraint: Constraint):
        """
        """
        if isinstance(constraint, Constraint) and constraint.canonicalize()[0].highest_order > 1:
            raise NotImplementedError("power exceed")

        return add_expr(self, constraint)

    return check


# def canonicalize(add_expr):
#     """
#     """
#
#     @wraps(add_expr)
#     def transform(self, expr: Expression):
#         if isinstance(expr, Expression):
#             op = expr.op
#             if op in OP_PAIRS:
#                 expr = (-(expr.with_op(''))).with_op(OP_PAIRS[op])
#         return add_expr(self, expr)
#
#     return transform


class Problem(object):
    __default_solvers__ = {
        "jet20.backend": jet20_default_backend_func,
    }

    def __init__(self, name: str = ""):
        self.name = name
        self._subject: Union[Expression, None] = None
        self._constraints: List[Constraint] = []
        self._variables: List[Variable] = []
        self._solver = {}  # 'solver_name': function
        self._solver.update(**self.__default_solvers__)

    @property
    def variables_count(self) -> int:
        """

        Returns: The numbers of variables this problem has contained

        """
        return len(self._variables)

    def variable(self, name: str, lb:Union[None, float] = None, ub:Union[None, float] = None) -> Variable:
        """Adding single variable with the name annotation.
        Args:
            name: The name of a variable

        Returns:
            A new variable
        """
        _var = Variable(self.variables_count, name)  # len(variables) exactly be equal with next variable's index
        self._variables += [_var]
        if lb is float:
            self._constraints += [Constraint(_var, lb, OP_GE)]
        if ub is float:
            self._constraints += [Constraint(_var, ub, OP_LE)]
        return _var

    def variables(self, symbols: str, lb: Union[None, List[float], float] = None,
                  ub: Union[None, List[float], float] = None) -> List[Variable]:
        """Adding a batch variables.
        Example: variables("x y z")
        Args:
            symbols: A set of in-order potential variable names, the separator is blank or comma or semicolon[ ,;]

        Returns:
            Variables, and their name is attached with per symbol
        """
        if lb is not (None, list, float) or ub is not (None, list, float):
            raise TypeError("bounds must be list of floats, float or None")

        _var_names = list(filter(None, re.split("[ ,;]", symbols)))

        if lb is list and len(_var_names) != len(lb):
            raise ValueError("mismatch length of lower bounds vector and variables vector")
        if ub is list and len(_var_names) != len(ub):
            raise ValueError("mismatch length of upper bounds vector and variables vector")

        _vars = []
        for i, symbol in enumerate(_var_names):
            _var = Variable(self.variables_count, symbol)
            self._variables += [_var]
            _vars += [_var]
            if lb is not None:
                self._constraints +=[Constraint(_var, lb, OP_GE)] if lb is float else [Constraint(_var, lb[i])]
            if ub is not None:
                self._constraints += [Constraint(_var, ub, OP_LE)] if ub is float else [Constraint(_var, ub[i])]

        return _vars

    def minimize(self, expr: Union[Expression, Array]):
        """Add the object math expression of this problem for minimizing it's value.
        Args:
            expr: A instance of Expression.

        Returns:

        """
        if isinstance(expr, Expression):
            self._subject = expr
        elif isinstance(expr, Array) and expr.len == 1:
            self._subject = expr.array[0]
        else:
            raise TypeError("expr must be a Expression or a Array")

    def maximize(self, expr: Union[Expression, Array]):
        """Add the object math expression of this problem for maximizing it's value.
        Args:
            expr: A instance of Expression.

        Returns:

        """
        if isinstance(expr, Expression):
            self._subject = -expr
        elif isinstance(expr, Array) and expr.len == 1:
            self._subject = -expr.array[0]
        else:
            raise TypeError("expr must be a Expression or a Array")

    @assert_power
    def constraints(self, *constraints: Union[Array, Constraint]):
        """Add a constraint.
        Args:
            constraints: A instance of Constraint, insists of a left value, an operator and a right value.

        Returns:

        """
        for cons in constraints:
            if isinstance(cons, Constraint):
                self._constraints += [cons]
            elif isinstance(cons, Array):
                self._constraints += [con for con in cons.array if isinstance(con, Constraint)]

    @property
    def canonical(self):
        """Return the canonical form of this problem.
        Returns: (object express matrix, list of constraint tuples).
         [[1. 0.  0.  0. ]
          [0. 2.  0.  0.5]  [[5. 4.0 4.3 0. ]
          [0. 0.  0.  0.5]   [0. 3.2 0.  2.1]
        ( [0. 0.5 0.5 0. ]], [1. 2.  3.  4. ]], ["<","<=",...], [const1,const2,...])
                  ^                  ^                  ^                 ^
                  |                  |                  |                 |
                object          constraits             ops              consts
        """
        _obj = self._subject.core_mat  # TODO: 是否要去掉const
        exprs, ops = list(zip(*[con.canonicalize() for con in self._constraints]))  # unzip constraints, ops
        _constraints = np.stack([con.linear_complete_vector[:-1] for con in exprs])  # cut const off
        _ops = np.array(ops)
        _consts = np.array([-con.const for con in exprs])
        return _obj, _constraints, _ops, _consts

    def solve(self, name: str = "jet20.backend", *args, **kwargs):  # TODO: return type hint
        """Calling one of the registered solvers to solve problem.
        Args:
            name: One of the registered solvers's name.
            *args: Extra args depends on the solver
            **kwargs: Extra args depends on the solver

        Returns:

        """
        return self._solver[name](self, *args, **kwargs)

    def reg_solver(self, name: str, func: Callable):  # TODO:可否约束func签名
        """Register a solver, it will be called to solve the problem later.
        Args:
            name: Name annotation of this solver.
            func: A function implemented to solve the problem.

        Returns:

        """
        self._solver[name] = func
