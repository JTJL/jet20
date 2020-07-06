import torch
import numpy as np
from jet20.backend.constraints import Constraints
from jet20.backend.core.linear_solver import LUSolver,CholeskySolver,CGSolver
from jet20.backend.core.status import *
from jet20.backend.const import LINEAR,QUADRATIC
from jet20.backend.core.utils import NotImproving

import logging
logger = logging.getLogger(__name__)

    

def line_search(r_norm,x_v,search_dir,norm,alpha=0.1,beta=0.5,max_cnt=0):    
    x,v = x_v
    dx,dv = search_dir
    
    s = 1.0
    
    new_x = x + dx * s
    new_v = v + dv * s
    
    while r_norm(new_x,new_v) > (1-alpha*s) * norm and s > 0:
        s = beta * s
        new_x = x + dx * s
        new_v = v + dv * s
        
    return s


def solve_kkt(h2,A,r_dual,r_pri,n,d):
    p = n + d
    KKT = A.new_zeros((p,p))

    if not (isinstance(h2,float) and h2 == 0.0):
        KKT[:n,:n] = h2

    KKT[:n,n:] = A.T
    KKT[n:,:n] = A

    solver = LUSolver()
    _dir = -solver(KKT,torch.cat([r_dual,r_pri]))
        
    _dir_x,_dir_v = _dir[:n],_dir[n:]
    
    return _dir_x,_dir_v


def solve_kkt_fast(h2,A,r_dual,r_pri):
    h_pd = h2
    g = r_dual
    h = r_pri
    
    solver = CholeskySolver()
    hat = solver(h_pd,A.T)
    hg = solver(h_pd,g)
    s = -A @ hat
    _dir_v = LUSolver()(s,A @ hg - h)
    _dir_x = solver(h_pd,-(A.T @ _dir_v + g))
    
    return _dir_x,_dir_v

def primal_dual_interior_point_with_eq(x,obj,eq_cons=None,should_stop=None,u=10.0, tolerance=1e-3, constraint_tolerance=1e-3, alpha=0.1, beta=0.5, fast=False, verbose=False):
    from torch.autograd.functional import jacobian
    from torch.autograd.functional import hessian
    
    n = x.size(0)
    d = eq_cons.size()
    
    v = x.new_ones(d)
    
    def l(x,v):
        return obj(x) + eq_cons(x) @ v
        
    def residual(x,v):
        r_dual = jacobian(lambda x: l(x,v),x)
        r_pri = eq_cons(x)
        return r_dual,r_pri
        
    def r_norm(x,v):
        r_dual,r_pri = residual(x,v)
        norm = torch.cat([r_dual,r_pri]).norm(2)
        return norm
    
    def jacobian_(f,x):
        if f.type() == LINEAR:
            return f.A
        else:
            return jacobian(f,x)
        
    def hessian_(x,v):
        if eq_cons.type() == LINEAR and obj.type() == LINEAR:
            return 0.0
        else:
            return hessian(lambda x: l(x,v),x)
    
    should_stop = should_stop or []
    not_improving = NotImproving()
        
    while True:        
        r_dual,r_pri = residual(x,v)
        obj_value = obj(x)
        norm = torch.cat([r_dual,r_pri]).norm(2)
        
        if verbose:
            logger.info("obj:%s,r_pri:%s,r_dual:%s,norm:%s",obj_value,r_pri.norm(2),r_dual.norm(2),norm)
        if r_pri.norm(2) <= constraint_tolerance and r_dual.norm(2) <= constraint_tolerance and norm <= tolerance:
            return x, obj_value, OPTIMAL

        if not_improving(norm):
            return x, obj_value, SUB_OPTIMAL
        
        h2 = hessian_(x,v)
        A = jacobian_(eq_cons,x)
        
        if fast and not (isinstance(h2,float) and h2 == 0.0):
            _dir_x,_dir_v = solve_kkt_fast(h2,A,r_dual,r_pri)
        else:
            _dir_x,_dir_v = solve_kkt(h2,A,r_dual,r_pri,n,d)
        
        step = line_search(r_norm,(x,v),(_dir_x,_dir_v), norm, alpha=alpha, beta=beta)

        x = x + step * _dir_x
        v = v + step * _dir_v
        
        for ss in should_stop:
            if ss(x,obj_value,None):
                return x, obj_value, USER_STOPPED
