import torch
import numpy as np
from jet20.backend.constraints import Constraints
from jet20.backend.core.linear_solver import LUSolver,CholeskySolver,CGSolver
from jet20.backend.core.status import *
from jet20.backend.core.utils import NotImproving
from jet20.backend.const import LINEAR,QUADRATIC

import logging
logger = logging.getLogger(__name__)


def line_search(r_norm,le,x_l_v,search_dir,t,norm,alpha=0.1,beta=0.5):    
    x,l,v, = x_l_v
    dx,dl,dv = search_dir

    tmp = -l[dl<0] / dl[dl<0]
    if tmp.nelement() == 0:
        s_max = 1.0
    else:
        s_max = min(1.0,tmp.min())
    
    # s_max = min(1.0,(-l[dl<0] / dl[dl<0]).min())
    s = 0.99*s_max
    
    new_x = x + dx * s
    while not le.validate(new_x) and s > 0:
        s = beta * s
        new_x = x + dx * s
    
    new_x = x + dx * s
    new_l = l + dl * s
    new_v = v + dv * s
    
    while r_norm(new_x,new_l,new_v,t) > (1-alpha*s) * norm and s > 0:
        s = beta * s
        new_x = x + dx * s
        new_l = l + dl * s
        new_v = v + dv * s
        
    return s


def solve_kkt(h2,d_f,A,lambda_,f_x,r_dual,r_cent,r_pri,n,m,d):
    p = n + m + d
    KKT = d_f.new_zeros((p,p))

    if not (isinstance(h2,float) and h2 == 0.0):
        KKT[:n,:n] = h2

    KKT[:n,n:n+m] = d_f.T
    KKT[:n,n+m:] = A.T

    KKT[n:n+m,:n] = -lambda_.unsqueeze(-1) * d_f
    KKT[n:n+m,n:n+m] = -torch.diag(f_x)

    KKT[n+m:,:n] = A

    solver = LUSolver()
    _dir = -solver(KKT,torch.cat([r_dual,r_cent,r_pri]))
        
    _dir_x,_dir_lambda,_dir_v = _dir[:n],_dir[n:n+m],_dir[n+m:]
    
    return _dir_x,_dir_lambda,_dir_v


def solve_kkt_fast(h2,d_f,A,lambda_,f_x,r_dual,r_cent,r_pri):
    if isinstance(f_x,torch.DoubleTensor):
        f_x[f_x == 0] = 1e-16
    else:
        f_x[f_x == 0] = 1e-8
    
    _r_cent = (f_x ** -1) * r_cent
    _d_f = -(lambda_ / f_x).unsqueeze(-1) * d_f

    h_pd = h2 + d_f.T @ _d_f
    g = r_dual + d_f.T @ _r_cent
    h = r_pri
    
    solver = CholeskySolver()
    hat = solver(h_pd,A.T)
    hg = solver(h_pd,g)
    s = -A @ hat
    _dir_v = LUSolver()(s,A @ hg - h)
    _dir_x = solver(h_pd,-(A.T @ _dir_v + g))

    _dir_lambda =  _d_f @ _dir_x + _r_cent
    
    return _dir_x,_dir_lambda,_dir_v

def primal_dual_interior_point_with_eq_le(x,obj,le_cons=None,eq_cons=None,should_stop=None,u=10.0, tolerance=1e-3,constraint_tolerance=1e-3, alpha=0.1, beta=0.5, fast=False,verbose=False):
    from torch.autograd.functional import jacobian
    from torch.autograd.functional import hessian
    
    m = le_cons.size()
    n = x.size(0)
    d = eq_cons.size()
    
    lambda_ = x.new_ones(m)
    v = x.new_ones(d)
    
    def l(x,lambda_,v):
        return obj(x) + le_cons(x) @ lambda_ + eq_cons(x) @ v
        
    def residual(x,lambda_,v,t):
        f_x = le_cons(x)
        r_dual = jacobian(lambda x: l(x,lambda_,v),x)
        r_cent = -lambda_ * f_x - 1/t
        r_pri = eq_cons(x)
        return r_dual,r_cent,r_pri
        
    def r_norm(x,lambda_,v,t):
        r_dual,r_cent,r_pri = residual(x,lambda_,v,t)
        norm = torch.cat([r_dual,r_cent,r_pri]).norm(2)
        return norm
    
    def jacobian_(f,x):
        if  f.type() == LINEAR:
            return f.A
        else:
            return jacobian(f,x)
        
    def hessian_(x,lambda_,v):
        if le_cons.type() == LINEAR and eq_cons.type() == LINEAR and obj.type() == LINEAR:
            return 0.0
        else:
            return hessian(lambda x: l(x,lambda_,v),x)
        
    should_stop = should_stop or []
    not_improving = NotImproving()

    while True:
        f_x = le_cons(x)
        dual_gap = - f_x @ lambda_
        t = u * m / dual_gap
        
        r_dual,r_cent,r_pri = residual(x,lambda_,v,t)
        obj_value = obj(x)
        norm = torch.cat([r_dual,r_cent,r_pri]).norm(2)
        
        if verbose:
            logger.info("obj:%s,r_pri:%s,r_dual:%s,dual_gap:%s,norm:%s",obj_value,r_pri.norm(2),r_dual.norm(2),dual_gap,norm)

        if r_pri.norm(2) <= constraint_tolerance and r_dual.norm(2) <= constraint_tolerance and dual_gap <= tolerance:
            return x, obj_value, OPTIMAL

        if not_improving(norm):
            return x, obj_value, SUB_OPTIMAL

        if torch.isnan(obj_value):
            return x, obj_value, FAIELD

        
        h2 = hessian_(x,lambda_,v)
        d_f = jacobian_(le_cons,x)
        A = jacobian_(eq_cons,x)

        
        if fast:
            _dir_x,_dir_lambda,_dir_v = solve_kkt_fast(h2,d_f,A,lambda_,f_x,r_dual,r_cent,r_pri)
        else:
            _dir_x,_dir_lambda,_dir_v = solve_kkt(h2,d_f,A,lambda_,f_x,r_dual,r_cent,r_pri,n,m,d)

        step = line_search(r_norm,le_cons,(x,lambda_,v),(_dir_x,_dir_lambda,_dir_v), t, norm, alpha=alpha, beta=beta)

        x = x + step * _dir_x
        v = v + step * _dir_v
        lambda_ = lambda_ + step * _dir_lambda

        
        for ss in should_stop:
            if ss(x,obj_value,dual_gap):
                return x, obj_value, USER_STOPPED
