from jet20.frontend.expression import Expression, Constraint
from jet20.frontend.variable import Variable, Array
from jet20.frontend.const import *
from jet20.frontend.backends import jet20_default_backend_func
from typing import List, Union, Callable
from functools import wraps
import numpy as np
import re


def assert_power(add_expr):
    """
    """

    @wraps(add_expr)
    def check(self, *constraints: Constraint):
        """
        """
        for c in constraints:
            if isinstance(c, Constraint) and c.canonicalize()[0].highest_order > 1:
                raise NotImplementedError("power exceed")

        return add_expr(self, *constraints)

    return check


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

    def variable(self, name: str, lb: Union[None, float] = None, ub: Union[None, float] = None) -> Variable:
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
        # if type(lb) not in (None, list, float) or type(ub) not in (None, list, float):
        #     raise TypeError("bounds must be list of floats, float or None")

        _var_names = list(filter(None, re.split("[ ,;]", symbols)))

        if isinstance(lb, list) and len(_var_names) != len(lb):
            raise ValueError("mismatch length of lower bounds vector and variables vector")
        if isinstance(ub, list) and len(_var_names) != len(ub):
            raise ValueError("mismatch length of upper bounds vector and variables vector")

        if lb is None or isinstance(lb, (float, int)):
            lb = [lb] * len(_var_names)

        if ub is None or isinstance(ub, (float, int)):
            ub = [ub] * len(_var_names)

        _vars = Array()
        for symbol, lb, ub in zip(_var_names, lb, ub):
            _var = Variable(self.variables_count, symbol, lb, ub)
            _vars.append(_var)
            self._variables.append(_var)

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

        for _var in self._variables:
            if _var.lb is not None:
                self._constraints.append(Constraint(_var, _var.lb, OP_GE))
            if _var.ub is not None:
                self._constraints.append(Constraint(_var, _var.ub, OP_LE))

        _obj = self._subject.core_mat  # need cut const off here?
        exprs, ops = list(zip(*[con.canonicalize() for con in self._constraints]))  # unzip constraints, ops
        _constraints = np.stack(
            [con.expand_linear_vector(len(self._variables) + 1)[:-1] for con in exprs])  # cut const off
        _ops = np.array(ops)
        _consts = np.array([-con.const for con in exprs])
        return _obj, _constraints, _ops, _consts

    def solve(self, name: str = "jet20.backend", *args, **kwargs):
        """Calling one of the registered solvers to solve the problem., jet20.backend will be used by default.
   
        :param name: One of the registered solvers's name.
        :type name: str
        :param args: Extra args depends on the solver
        :param kwargs: Extra args depends on the solver
        :return: solution of the problem depends on the solver

        for jet20.backend following args can be used:

        :param x: initial solution of the problem
        :type x: list,numpy.ndarray
        :param opt_u: hyperparameters for interior point method
        :type opt_u: float
        :param opt_alpha: hyperparameters for line search
        :type opt_alpha: float
        :param opt_beta: hyperparameters for line search
        :type opt_beta: float
        :param opt_tolerance: objective value tolerance
        :type opt_tolerance: float
        :param opt_constraint_tolerance: feasibility tolerance
        :type opt_constraint_tolerance: float
        :param rouding_precision: rouding precision
        :type rouding_precision: int
        :param force_rouding: whether force rounding
        :type rouding_precision: bool
        :return: solution of the problem
        :rtype: Solution

        """
        return self._solver[name](self, *args, **kwargs)

    def reg_solver(self, name: str, func: Callable):
        """Register a solver, it will be called to solve the problem later.
        Args:
            name: Name annotation of this solver.
            func: A function implemented to solve the problem.

        Returns:

        """
        self._solver[name] = func
