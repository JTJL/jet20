from jet20.frontend.expression import Expression
from jet20.frontend.variable import Variable
from jet20.frontend.const import *
from jet20.frontend.backends import jet20_default_backend_func
from typing import List, Union, Callable
from functools import wraps
import numpy as np
import re


# TODO:提供with处理exception

def assert_not_constraint(add_expr):
    """
    """

    @wraps(add_expr)
    def check(self, expr: Expression):
        """
        """
        if isinstance(expr, Expression) and expr.is_constraint:
            raise NotImplementedError("unsupported put a constraint as object")

        return add_expr(self, expr)

    return check


def assert_power(add_expr):
    """
    """

    @wraps(add_expr)
    def check(self, expr: Expression):
        """
        """
        if isinstance(expr, Expression) and expr.highest_order > 1:
            raise NotImplementedError("power exceed")

        return add_expr(self, expr)

    return check


def canonicalize(add_expr):
    """
    """

    @wraps(add_expr)
    def transform(self, expr: Expression):
        if isinstance(expr, Expression):
            op_pairs = {OP_GT: OP_LT, OP_GE: OP_LE}  # {">":"<", ">=":"<="}
            op = expr.op
            if op in op_pairs:
                expr = (-(expr.with_op(''))).with_op(op_pairs[op])
        return add_expr(self, expr)

    return transform


class Problem(object):
    __default_solvers__ = {
        "jet20.backend": jet20_default_backend_func,
    }

    def __init__(self,name: str = ""):
        self.name = name
        self._object: Union[Expression, None] = None
        self._constraints: List[Expression] = []
        self._variables: List[Variable] = []
        self._solver = {}  # 'solver_name': function
        self._solver.update(**self.__default_solvers__)

    @property
    def variables_count(self) -> int:
        """

        Returns: The numbers of variables this problem has contained

        """
        return len(self._variables)

    def variable(self, name: str):
        """Adding single variable with the name annotation.
        Args:
            name: The name of a variable

        Returns:
            A new variable
        """
        _var = Variable(self.variables_count, name)  # len(variables) exactly be equal with next variable's index
        self._variables += [_var]
        return _var

    def variables(self, symbols: str, lb:Union[None,List[float],float]=None, ub:Union[None,List[float],float]=None):
        """Adding a batch variables.
        Example: variables("x y z")
        Args:
            symbols: A set of in-order potential variable names, the separator is blank or comma or semicolon[ ,;]

        Returns:
            Variables, and their name is attached with per symbol
        """
        _vars = []
        for symbol in list(filter(None, re.split("[ ,;]", symbols))):
            _var = Variable(self.variables_count, symbol)
            self._variables += [_var]
            _vars += [_var]
        return _vars

    @assert_not_constraint
    def minimize(self, expr: Expression):
        """Add the object math expression of this problem for minimizing it's value.
        Args:
            expr: A instance of Expression.

        Returns:

        """
        self._object = expr

    @assert_not_constraint
    def maximize(self, expr: Expression):
        """Add the object math expression of this problem for maximizing it's value.
        Args:
            expr: A instance of Expression.

        Returns:

        """
        self._object = -expr  # transform max function to canonical type(min type)

    @assert_power
    @canonicalize  # x+y > 1 to -x-y <= 1
    def constraint(self, expr: Expression):
        """Add a math expression as a constraint.
        Args:
            expr: A instance of Expression, it must be in constraint form(contains a comparison operator).

        Returns:

        """
        if not expr:
            raise ValueError("constraint is empty")
        if not isinstance(expr, Expression):
            raise TypeError("constraint must Expression type")
        if not expr.is_constraint:
            raise TypeError("expression is not a constraint(A constraint must contained an constraint operator)")

        # resize constraint to same shape and append in
        self._constraints += [expr.expand(self.variables_count + 1)]

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
        _obj = self._object.core_mat  # TODO: 是否要去掉const
        _constraints = np.stack([con.linear_complete_vector[:-1] for con in self._constraints])  # cut const off
        _ops = np.array([con.op for con in self._constraints])
        _consts = np.array([-con.const for con in self._constraints])
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
