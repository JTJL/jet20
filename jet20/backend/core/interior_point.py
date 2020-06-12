import torch

import logging
logger = logging.getLogger(__name__)


def line_search(f,x,search_dir,g,alpha=0.3,beta=0.5,max_cnt=0):
    t = 1.0
    f_delta_x = f(x + search_dir * t)
    f_x = f(x)
    _delta = torch.dot(search_dir,g)
    _c = 0
    
    while torch.isnan(f_delta_x) or f(x + search_dir * t) > f_x + alpha * t * _delta:
        t = beta * t
        f_delta_x = f(x + search_dir * t)
        _c += 1
        if max_cnt > 0 and _c >= max_cnt:
            return 0.0, True
        
    return t,False
    

def newton(x,obj,le_cons = None,eq_cons=None,should_stop=None,t = 1.0,tolerance=1e-10, max_cnt=0, alpha=0.3, beta=0.5):
    _c = 0
    
    if le_cons is not None:
        def f(x):
            return t * obj(x) - torch.log(-1 * le_cons(x)+1e-15).sum()
    else:
        def f(x):
            return obj(x)
    
    while True:
        jacobian = torch.autograd.functional.jacobian(f,x)

        hessian = torch.autograd.functional.hessian(f,x)
        try:
            hessian_inverse = hessian.inverse()
        except RuntimeError as e:
            #todo: Determine whether it is optimal and switch to gradient desent if needed
            #for now: just skip
            logger.error(e,exc_info=True)
            # logger.warn("no inverse for hessian matrix")
            if not str(e).startswith("inverse_"):
                raise e
            return x,False
    
        if eq_cons is not None:
            A = eq_cons.A
            hat = torch.mm(hessian_inverse,A.T)
            nd = torch.mv(hessian_inverse,jacobian)
            s = - torch.mm(A,hat)
            w = torch.mv(torch.mm(s.inverse(),A),nd)
            _dir = -1 * torch.mv(hessian_inverse,jacobian + torch.mv(A.T,w))
        else:
            _dir = -torch.mv(hessian_inverse,jacobian)

        if torch.isnan(_dir.sum()):
            logging.warn("search dir has nan...")
            return x,False

        step,timeout = line_search(f,x,_dir,jacobian)
        if timeout:
            logging.warn("time out while line search...")
            return x,True

        x = x + step * _dir

        if torch.dot(jacobian,_dir) < tolerance or (should_stop is not None and should_stop(x)):
            return x,False

        _c += 1      
        if max_cnt > 0 and _c >= max_cnt:
            return x,True


def interior_point(x,obj,le_cons = None,eq_cons=None,should_stop=None, t=1.0, u=10.0, tolerance=1e-10, **kwargs):
    if le_cons is None:
        m = 0.
    else:
        m = le_cons.A.size(0)
        
    while True:
        x,timeout = newton(x,obj,le_cons,eq_cons,should_stop,t,tolerance,**kwargs)
        if timeout:
            logger.warn('time out while running newton method..')
        if m / t < tolerance or (should_stop is not None and should_stop(x)) :
            return x
        t = t * u