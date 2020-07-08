import torch
import numpy as np
from jet20.backend.constraints import Constraints
from jet20.backend.core.linear_solver import LUSolver,CholeskySolver,CGSolver
from jet20.backend.core.status import *
from jet20.backend.const import LINEAR,QUADRATIC
from jet20.backend.core.utils import NotImproving

import logging
logger = logging.getLogger(__name__)



def line_search(r_norm,le,x_l,search_dir,t,norm,alpha=0.1,beta=0.5):    
    x,l = x_l
    dx,dl = search_dir

    tmp = -l[dl<0] / dl[dl<0]
    if tmp.nelement() == 0:
        s_max = 1.0
    else:
        s_max = min(1.0,tmp.min())
    
    s = 0.99*s_max
    
    new_x = x + dx * s
    while not le.validate(new_x) and s > 0:
        s = beta * s
        new_x = x + dx * s
    
    new_x = x + dx * s
    new_l = l + dl * s
    
    while r_norm(new_x,new_l,t) > (1-alpha*s) * norm and s > 0:
        s = beta * s
        new_x = x + dx * s
        new_l = l + dl * s
        
    return s


def solve_kkt_fast(h2,d_f,lambda_,f_x,r_dual,r_cent):
    if isinstance(f_x,torch.DoubleTensor):
        f_x[f_x == 0] = 1e-16
    else:
        f_x[f_x == 0] = 1e-8

    _r_cent = (f_x ** -1) * r_cent
    _d_f = -(lambda_ / f_x).unsqueeze(-1) * d_f
    
    h_pd = h2 + d_f.T @ _d_f
    g = r_dual + d_f.T @ _r_cent
    
    solver = CholeskySolver()
    _dir_x = -solver(h_pd,g)
    _dir_lambda =  _d_f @ _dir_x + _r_cent
    
    return _dir_x,_dir_lambda

def solve_kkt(h2,d_f,lambda_,f_x,r_dual,r_cent,n,m):
    p = n+m
    KKT = d_f.new_zeros((p,p))

    if not (isinstance(h2,float) and h2 == 0.0):
        KKT[:n,:n] = h2

    KKT[:n,n:n+m] = d_f.T

    KKT[n:n+m,:n] = -lambda_.unsqueeze(-1) * d_f
    KKT[n:n+m,n:n+m] = -torch.diag(f_x)
    
    solver = LUSolver()
    _dir = -solver(KKT,torch.cat([r_dual,r_cent]))
    _dir_x,_dir_lambda = _dir[:n],_dir[n:n+m]
    
    return _dir_x,_dir_lambda
    
    
    
def primal_dual_interior_point_with_le(x,obj,le_cons=None,should_stop=None,u=10.0, tolerance=1e-3, constraint_tolerance=1e-3,
             alpha=0.1, beta=0.5, fast=False,verbose=False,duals=None):
    from torch.autograd.functional import jacobian
    from torch.autograd.functional import hessian
    
    m = le_cons.size()
    n = x.size(0)
    
        
    if duals is None:
        lambda_ = x.new_ones(m)
    else:
        lambda_ = duals

    u = 10
    
    def l(x,lambda_):
        return obj(x) + le_cons(x) @ lambda_
        
    def residual(x,lambda_,t):
        f_x = le_cons(x)
        r_dual = jacobian(lambda x: l(x,lambda_),x)
        r_cent = -lambda_ * f_x - 1/t
        return r_dual,r_cent
        
    def r_norm(x,lambda_,t):
        r_dual,r_cent = residual(x,lambda_,t)
        norm = torch.cat([r_dual,r_cent]).norm(2)
        return norm
    
    def jacobian_(f,x):
        if f.type() == LINEAR:
            return f.A
        else:
            return jacobian(f,x)
        
        
    def hessian_(x,lambda_):
        if le_cons.type() == LINEAR and obj.type() == LINEAR:
            return 0.0
        else:
            return hessian(lambda x: l(x,lambda_),x)

    should_stop = should_stop or []
    not_improving = NotImproving()

    while True:
        f_x = le_cons(x)
        dual_gap = - f_x @ lambda_
        t = u * m / dual_gap
        
        r_dual,r_cent = residual(x,lambda_,t)
        obj_value = obj(x)
        norm = torch.cat([r_dual,r_cent]).norm(2)

        if verbose:
            logger.info("obj:%s,r_dual:%s,r_cent:%s,norm:%s",obj_value,r_dual.norm(2),r_cent.norm(2),norm)
            
        if r_dual.norm(2) <= constraint_tolerance and dual_gap <= tolerance:
            return x, obj_value, OPTIMAL, lambda_

        if not_improving(norm):
            return x, obj_value, SUB_OPTIMAL, lambda_
        
        if torch.isnan(obj_value):
            return x, obj_value, FAIELD, lambda_
        
        h2 = hessian_(x,lambda_)
        d_f = jacobian_(le_cons,x)
        
        if fast:
            _dir_x,_dir_lambda = solve_kkt_fast(h2,d_f,lambda_,f_x,r_dual,r_cent)
        else:
            _dir_x,_dir_lambda = solve_kkt(h2,d_f,lambda_,f_x,r_dual,r_cent,n,m)
    
        
        step = line_search(r_norm,le_cons,(x,lambda_),(_dir_x,_dir_lambda), t, norm, alpha=alpha, beta=beta)
        x = x + step * _dir_x
        lambda_ = lambda_ + step * _dir_lambda
        
        for ss in should_stop:
            if ss(x,obj_value,dual_gap):
                return x, obj_value, USER_STOPPED, lambda_