
import torch

from jet20.backend.plugins import Plugin
from jet20.backend.constraints import *
from jet20.backend.obj import *
from jet20.backend.core import solve
from jet20.backend.solver import Problem,USER_STOPPED,SUB_OPTIMAL,OPTIMAL

import logging
logger = logging.getLogger(__name__)


class LinearDependent(Exception):pass

class EnsureEqFeasible(Plugin):

    def find_feasible(self,eq,config):
        A = eq.A
        b = eq.b

        u,_lambda,v = A.svd()
        if (_lambda < 1e-8).float().sum() > 0:
            raise LinearDependent("linear dependent in eq constraints..")
        
        x = v @ torch.diag(_lambda**-1) @  u.T @ b

        # if not eq.validate(x,config.opt_constraint_tolerance):
        #     logger.debug("delta:%s",A @ v - b)
        #     raise EqConstraitConflict("confilct in eq constraints")

        return x

    def preprocess(self,p,x,config):
        if not p.eq:
            return p,x
        
        if x is None:
            return p,self.find_feasible(p.eq,config)

        if not p.eq.validate(x):
            logger.warn("x is not a feasible solution, eq constraints not satisfied")
            return p,self.find_feasible(p.eq,config)
        
        return p,x


    def postprocess(self,p,x,config):
        if p.eq and not p.eq.validate(x,config.opt_constraint_tolerance):
            raise EqConstraitConflict("confilct in eq constraints")
        
        return p,x
             


class EnsureLeFeasible(Plugin):

    def find_feasible(self,p,x,config):
        if p.eq:
            if p.eq.type() != LINEAR:
                raise NotImplementedError("non linear constrait not supported")

            _A = torch.cat([p.eq.A.new_zeros(p.eq.A.size(0)).unsqueeze(-1),p.eq.A],dim=1)
            eq = LinearEqConstraints(_A,p.eq.b)
        else:
            eq = None
        
        _A = torch.cat([-1 * p.le.A.new_ones(p.le.A.size(0)).unsqueeze(-1),p.le.A],dim=1)
        le = LinearLeConstraints(_A,p.le.b)

        if x is None:
            x = _A.new_ones(p.n)

        s = (torch.mv(p.le.A,x) - p.le.b).max()+1e-3
        x = torch.cat([s.unsqueeze(0),x])

        def f(x):
            return x[0]

        def should_stop(x,obj_value,dual_gap):
            return obj_value < 0


        obj = LambdaObjective(LINEAR,f)

        _p = Problem([],obj,le,eq)
        _p.float()
        x = x.float()

        x,obj_value,status = solve(_p,x,config,fast=True,should_stops=[should_stop])
        if status == SUB_OPTIMAL and obj_value > 0:
            _p.double()
            x = x.double()
            x,obj_value,status = solve(_p,x,config,fast=False,should_stops=[should_stop])


        if status == USER_STOPPED or obj_value <= 0:
            return x[1:]
        else:
            raise LeConstraitConflict("conflict in le constraints")
        
    
    def preprocess(self,p,x,config):
        if not p.le:
            return p,x
        
        if x is None:
            return p,self.find_feasible(p,x,config)

        if not p.le.validate(x):
            logger.warn("x is not a feasible solution, le constraints not satisfied")
            return p,self.find_feasible(p,x,config)

        return p,x


    def postprocess(self,p,x,config):
        if p.le and not p.le.validate(x):
            raise LeConstraitConflict("conflict in le constraints")
        
        return p,x