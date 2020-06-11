

from sympy import symarray,Matrix
from sympy.solvers.solveset import linsolve
import torch
import copy
from jet20.backend.plugins import Plugin
from jet20.backend.constraints import *


def cross(_vars):
    for x in _vars:
        for y in _vars:
            yield x*y


class Simpify(Plugin):
    def __init__(self):
        self.origin = None


    def solve_equitions(self,A,b,n):
        x = Matrix(symarray('x', n, commutative=False))                
        ret = linsolve((A,b), *x) 
        if ret.is_empty:
            raise EqConstraitConflict("confilct in eq constraints")
        ret = list(ret)[0]
        return x,Matrix(ret)


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
        return [ float(coeffs[var]) for var in free_vars ],coeffs[1]

    def quadratic_expr_var_subs(self,Q,b,c,rhs,free_vars):
        coeffs = (rhs.T @ Q @ rhs - b.dot(rhs) + c).expand().as_coefficients_dict()
        q = [coeffs[x] for x in cross(free_vars)]
        b = [coeffs[x] for x in free_vars]
        c = coeffs[1]
        return q,b,c


    def preprocess(self,p,_x,config):
        if not p.eq:
            return p,_x

        xs, rhs = self.solve_eq(p.eq.A,p.eq.b,len(p.free_vars))

        _frees = []
        for var,r in zip(p.free_vars,rhs):
            if r.is_constant():
                p.fix_vars[var] = float(r)
            else:
                _frees.append(var)

        


        







        

        return eq.A.new(ret)

        


    def preprocess(self,p,x,config):
        pass
    

    