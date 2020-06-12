
import torch

from jet20.backend.plugins import Plugin
from jet20.backend.constraints import *
from jet20.backend.core import interior_point

import logging
logger = logging.getLogger(__name__)


class EnsureEqFeasible(Plugin):

    def find_feasible(self,eq,config):
        A = eq.A
        b = eq.b

        u,_lambda,v = A.svd()
        x = v @ torch.diag(_lambda**-1) @  u.T @ b

        if not eq.validate(x,config.eq_constraint_tolerance):
            logger.debug("delta:%s",A @ v - b)
            raise EqConstraitConflict("confilct in eq constraints")

        return x

    def preprocess(self,p,x,config):
        if not p.eq:
            return p,x
        
        if x is None:
            return p,self.find_feasible(p.eq,config)

        if not p.eq.validate(x):
            logger.warn("x is not a feasible solution, eq constraints not satisfied")
            return p,self.find_feasible(p.eq,config)


    def postprocess(self,p,x,config):
        if p.eq and not p.eq.validate(x,config.eq_constraint_tolerance):
            raise EqConstraitConflict("confilct in eq constraints")
        
        return p,x
             


class EnsureLeFeasible(Plugin):

    def find_feasible(self,p,x,config):
        if p.eq:
            _A = torch.cat([p.eq.A.new_zeros(p.eq.A.size(0)).unsqueeze(-1),p.eq.A],dim=1)
            eq = LinearEqConstraints(_A,p.eq.b)
        else:
            eq = None

        if x is None:
            x = p.le.A.new_ones(p.le.A.size(1))
        
        _A = torch.cat([-1 * p.le.A.new_ones(p.le.A.size(0)).unsqueeze(-1),p.le.A],dim=1)
        le = LinearLeConstraints(_A,p.le.b)

        s = (torch.mv(p.le.A,x) - p.le.b).max()+1
        x = torch.cat([s.unsqueeze(0),x])

        def obj(x):
            return x[0]

        def should_stop(x):
            return x[0] < 0

        x = interior_point(x,obj,le,eq,should_stop,**config.get_namespace("opt"))

        if x[0] > 0:
            raise LeConstraitConflict("conflict in le constraints")

        return x[1:]

    def preprocess(self,p,x,config):
        if not p.le:
            return p,x
        
        if x is None:
            return p,self.find_feasible(p,x,config)

        if not p.le.validate(x):
            logger.warn("x is not a feasible solution, le constraints not satisfied")
            return p,self.find_feasible(p,x,config)


    def postprocess(self,p,x,config):
        if p.le and not p.le.validate(x):
            raise LeConstraitConflict("conflict in le constraints")
        
        return p,x