
import torch
import time
import copy
from jet20.backend.constraints import *
from jet20.backend.obj import *
from jet20.backend.config import *
from jet20.backend.core import solve,OPTIMAL,SUB_OPTIMAL,USER_STOPPED


import logging
logger = logging.getLogger(__name__)
  

class Solution(object):
    def __init__(self,x,_vars,obj_value,status,duals):
        self.status = status
        self.obj_value = obj_value
        self.vars = _vars
        self.x = x
        self.duals = None

    def __str__(self):
        return "obj_value: %s vars:%s" % (self.obj_value,self.vars)

    __repr__ = __str__


class Problem(object):
    def __init__(self,_vars,obj,le_cons=None,eq_cons=None):
        self.obj = obj
        self.le = le_cons
        self.eq = eq_cons
        self.vars = _vars
        self.n = len(_vars)


    @classmethod
    def from_numpy(cls,_vars,obj=None,le=None,eq=None,device=torch.device("cpu"),dtype=torch.float64):

        def convert(x):
            if x is not None:
                if isinstance(x,torch.Tensor):
                    return x.type(dtype).to(device)
                else:
                    return torch.tensor(x,dtype=dtype,device=device)
            else:
                return None

        if obj is not None:
            obj_Q,obj_b,obj_c = [convert(x) for x in obj]
            if obj_Q is not None:
                obj = QuadraticObjective(obj_Q,obj_b,obj_c)
            elif obj_b is not None:
                obj = LinearObjective(obj_b,obj_c)

        if le is not None:
            le_A,le_b = [convert(x) for x in le]
            if le_b.ndim == 2 and le_b.size(0) == 1:
                le_b = le_b.squeeze(0)
            le = LinearLeConstraints(le_A,le_b)
        
        if eq is not None:
            eq_A,eq_b = [convert(x) for x in eq]
            if eq_b.ndim == 2 and eq_b.size(0) == 1:
                eq_b = eq_b.squeeze(0)
            eq = LinearEqConstraints(eq_A,eq_b)

        return cls(_vars,obj,le,eq)


    def float(self):
        if self.le is not None:
            le = self.le.float()
        else:
            le = None

        if self.eq is not None:
            eq = self.eq.float()
        else:
            eq = None

        obj = self.obj.float()
        return self.__class__(self.vars,obj,le,eq)


    def double(self):
        if self.le is not None:
            le = self.le.double()
        else:
            le = None

        if self.eq is not None:
            eq = self.eq.double()
        else:
            eq = None

        obj = self.obj.double()

        return self.__class__(self.vars,obj,le,eq)

    def to(self,device):
        if self.le is not None:
            self.le.to(device)
        else:
            le = None

        if self.eq is not None:
            self.eq.to(device)
        else:
            eq = None

        obj = self.obj.to(device)
        return self.__class__(self.vars,obj,le,eq)


    def build_solution(self,x,obj_value,status,duals):
        _vars = { var: v.item() for var,v in zip(self.vars,x)}
        return Solution(x.cpu().numpy(),_vars,obj_value.item(),status,duals)


class Solver(object):
    def __init__(self):
        self.pres = []
        self.posts = []

    def solve(self,p,config,x=None):
        for pre in self.pres:
            start = time.time()
            p,x = pre.preprocess(p,x,config)
            logger.debug("preprocessing name:%s, time used:%s",pre.name(),time.time()-start)

        if x is None:
            x = torch.zeros(p.n).float().to(config.device)

        start = time.time()
        p_f32 = p.float()
        x = x.float()
        x,_,status,duals = solve(p_f32,x,config,fast=True)
        logger.debug("fast mode, time used:%s",time.time()-start)
        x = x.double()
        if isinstance(duals,(tuple,list)):
            duals = [d.double() for d in duals]
        else:
            duals = duals.double()

        if status == SUB_OPTIMAL:
            start = time.time()
            # p = p.double()
            x,_,status,duals = solve(p,x,config,fast=True,duals=duals)
            logger.debug("fast-precision mode, time used:%s",time.time()-start)

        if status == SUB_OPTIMAL:
            start = time.time()
            x,_,status,duals = solve(p,x,config,fast=False,duals=duals)
            logger.debug("precision mode, time used:%s",time.time()-start)


        if status != OPTIMAL:
            logger.warning("optimal not found, status:%s",status)

        for post in self.posts:
            start = time.time()
            p,x = post.postprocess(p,x,config)
            logger.debug("postprocessing name:%s, time used:%s",post.name(),time.time()-start)

        return p.build_solution(x,p.obj(x),status,duals)


    def register_pres(self,*pres):
        self.pres.extend(pres)
    
    def register_posts(self,*posts):
        self.posts.extend(posts)







        
        


