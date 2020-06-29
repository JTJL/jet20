
import numpy as np
from sympy import symarray,Matrix
from sympy.solvers.solveset import linsolve
import torch
import copy
from jet20.backend.plugins import Plugin
from jet20.backend.constraints import *
from jet20.backend.obj import *


import logging
logger = logging.getLogger(__name__)

def cross(_vars):
    for x in _vars:
        for y in _vars:
            yield x*y

def to_matrix(x):
    return Matrix(x.cpu())


class Simpify(Plugin):
    def __init__(self):
        self.vars_mapping = None
        self.free_vars = None


    def solve_equitions(self,A,b,x):               
        ret = linsolve((A,b), *x) 
        if ret.is_empty:
            raise EqConstraitConflict("confilct in eq constraints")
        ret = list(ret)[0]
        return Matrix(ret)

    def linear_equitions_var_subs(self,A,b,rhs,free_vars):
        equitions = A @ rhs - b
        rows = []
        cs = []

        for eq in equitions:
            coeffs = eq.expand().as_coefficients_dict()
            rows.append([ float(coeffs[var]) for var in free_vars])
            cs.append(-1 * float(coeffs[1]))

        return rows,cs

    def linear_expr_var_subs(self,b,c,rhs,free_vars):
        coeffs = (b.dot(rhs)+c).expand().as_coefficients_dict()
        return [ float(coeffs[var]) for var in free_vars ],float(coeffs[1]) 

    def quadratic_expr_var_subs(self,Q,b,c,rhs,free_vars):
        coeffs = (rhs.T @ Q @ rhs - b.dot(rhs) + c).expand().as_coefficients_dict()
        q = [float(coeffs[x]) for x in cross(free_vars)]
        b = [float(coeffs[x]) for x in free_vars]
        c = float(coeffs[1])
        return q,b,c

    def new_x(self,old_free_vars,new_free_vars,xs):
        if xs is None:
            return xs
        return [xs[np.where(old_free_vars == var)][0] for var in new_free_vars]


    def preprocess(self,p,_x,config):
        if not p.eq:
            return p,_x

        free_vars = symarray('x', p.n, commutative=False)
        free_vars = Matrix(free_vars)
        rhs = self.solve_equitions(to_matrix(p.eq.A),to_matrix(p.eq.b),free_vars)

        p.eq = None
        self.vars_mapping = rhs
        new_free_vars = list(rhs.free_symbols)
        new_x = self.new_x(free_vars,new_free_vars,_x)
        self.free_vars = new_free_vars
    
        new_free_vars = Matrix(new_free_vars)
        A,b = self.linear_equitions_var_subs(to_matrix(p.le.A),to_matrix(p.le.b),rhs,new_free_vars)
        p.le.A = p.le.A.new(A)
        p.le.b = p.le.b.new(b)

        if isinstance(p.obj,LinearObjective):
            b,c = self.linear_expr_var_subs(to_matrix(p.obj.b),float(p.obj.c),rhs,new_free_vars)
            p.obj.b = p.obj.b.new(b)
            p.obj.c = c

        if isinstance(p.obj,QuadraticObjective):
            A,b,c = self.quadratic_expr_var_subs(to_matrix(p.obj.A),to_matrix(p.obj.b),float(p.obj.c),rhs,new_free_vars)
            p.obj.A = p.obj.A.new(A)
            p.obj.b = p.obj.b.new(b)
            p.obj.c = c

        return p,new_x


    def postprocess(self,p,X,config):
        if not p.eq:
            return p,X
            
        logger.debug("vars_mapping:%s",self.vars_mapping)
        new_x = self.vars_mapping.subs({v:x for v,x in zip(self.free_vars,X)})
        new_x = X.new([_x.n() for _x in new_x])
        return p,new_x
    

    