
import torch
import time
import copy
from jet20.backend.constraints import *
from jet20.backend.obj import *
from jet20.backend.core.interior_point import interior_point


import logging
logger = logging.getLogger(__name__)

class Config(object):
    __default_config__ = {
        "opt_tolerance" : 1e-10,
        "opt_u" : 10.0,
        "opt_max_cnt" : 0,
        "opt_alpha" : 0.3,
        "opt_beta" : 0.5,
        "eq_constraint_tolerance": 1e-3,
        "scaling_desired_scale": 1,
        "rouding_precision": 2,
        "precision": "double",
        "device": "cuda",
    }

    def __init__(self,**kwargs):
        self.__dict__.update(**self.__default_config__)
        self.__dict__.update(kwargs)
        self.device = torch.device(self.device)

    def get_namespace(self,namespace):
        rv = {}
        for k, v in self.__dict__.items():
            if not k.startswith(namespace):
                continue
            key = k[len(namespace)+1 :]
            rv[key] = v
        return rv
  

class Solution(object):
    def __init__(self,obj_value,vars,x):
        self.obj_value = obj_value
        self.vars = vars
        self.x = x

    def __str__(self):
        return "obj_value: %s vars:%s" % (self.obj_value,self.x)


class Problem(object):
    def __init__(self,_vars,obj,le_cons=None,eq_cons=None):
        self.obj = obj
        self.le = le_cons
        self.eq = eq_cons
        self.vars = _vars
        self.n = len(_vars)


    def build_solution(self,x):
        obj_value = self.obj(x).item()
        _vars = { var: v.item() for var,v in zip(self.vars,x)}
        return Solution(obj_value,_vars,x.cpu().numpy())


class Solver(object):
    def __init__(self):
        self.pres = []
        self.posts = []

    def solve(self,p,config,x=None):
        origin_p = copy.deepcopy(p)

        for pre in self.pres:
            start = time.time()
            p,x = pre.preprocess(p,x,config)
            logger.debug("preprocessing name:%s, time used:%s",pre.name(),time.time()-start)

        start = time.time()
        x = interior_point(x,p.obj,p.le,p.eq,**config.get_namespace("opt"))
        logger.debug("caculation, time used:%s",time.time()-start)

        p = origin_p
        for post in self.posts:
            start = time.time()
            p,x = post.postprocess(p,x,config)
            logger.debug("postprocessing name:%s, time used:%s",post.name(),time.time()-start)

        return p.build_solution(x)


    def register_pres(self,*pres):
        self.pres.extend(pres)
    
    def register_posts(self,*posts):
        self.posts.extend(posts)







        
        


    

