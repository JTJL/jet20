import torch

import logging
logger = logging.getLogger(__name__)
from jet20.backend.core.linear_solver import LUSolver,CholeskySolver,CGSolver

    
def line_search(f,x,search_dir,nt_delta,alpha=0.1,beta=0.5):
    t = 1.0
    f_x = f(x)
    
    new_x = x + search_dir * t
    f_delta_x = f(new_x)
    
    while torch.isnan(f_delta_x) or f_delta_x > f_x + alpha * t * nt_delta:
        t = beta * t
        new_x = x + search_dir * t
        f_delta_x = f(new_x)
        
    return t,False
    

def newton(x,obj,le_cons=None,eq_cons=None,t = 1.0,tolerance=1e-4, alpha=0.1, beta=0.5):

    if le_cons is not None:
        def f(x):
            return t * obj(x) - torch.log(-1 * le_cons(x)+1e-8).sum()
    else:
        def f(x):
            return obj(x)
        
    if eq_cons is not None:
        v = eq_cons.A.zeros(eq_cons.A.size(0))
    
    while True:
        jacobian = torch.autograd.functional.jacobian(f,x)
        hessian = torch.autograd.functional.hessian(f,x)

        solver = CholeskySolver()

        if eq_cons is not None:
            A = eq_cons.A
            h = eq_cons(x)
            hat = solver(hessian,A.T)
            hg = solver(hessian,jacobian)
            s = -A @ hat
            w = LUSolver()(s,A @ hg - h)
            _dir = solver(hessian,-(A.T @ w + jacobian))

        else:
            _dir = -solver(hessian,jacobian)

        nt_delta = jacobian @ -_dir
        if nt_delta < 2 * tolerance:
            return x
        
        step,_ = line_search(f, x, _dir, nt_delta, alpha=alpha, beta=beta)

        x = x + step * _dir


def interior_point(x,obj,le_cons = None,eq_cons=None,should_stop=None, t=1.0, u=10.0, tolerance=1e-3, **kwargs):
    if le_cons is None:
        m = 0.
    else:
        m = le_cons.A.size(0)

    should_stop = should_stop or []
    
    while True:
        gap = m / t
        x = newton(x,obj,le_cons,eq_cons,t,max(gap,tolerance),**kwargs)
        obj_value = obj(x)
        logger.debug("obj:%s,dual_gap:%s",obj_value,gap)
        if m / t < tolerance:
            return x, obj_value, True
                   
        for ss in should_stop:
            if ss(x,obj_value,gap):
                return x, obj_value, False

        t = t * u