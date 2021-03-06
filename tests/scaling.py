

import torch
from jet20.backend.plugins import Plugin
from jet20.backend.obj import LinearObjective,QuadraticObjective

import logging
logger = logging.getLogger(__name__)



def calc_scale(x):
    x = x.abs()
    x = x[x>0]
    return torch.log10(x).floor().mean()

def scale_by(x,diff):
    return x * 10 ** diff

class Scaling(Plugin):

    def __init__(self):
        pass

    def scale_m(self,a,desired_scale):
        s = calc_scale(a)
        a = scale_by(a,desired_scale-s)
        return a

    def scale_row(self,a,b,desired_scale):
        s = calc_scale(a)
        a = scale_by(a,desired_scale-s)
        b = scale_by(b,desired_scale-s)
        return a,b

    def scale(self,A,b,desired_scale):
        assert A.size(0) == b.size(0)
        newA,newb = [],[]

        for x,y in zip(A,b):
            x,y = self.scale_row(x,y,desired_scale)
            newA.append(x)
            newb.append(y)

        return torch.stack(newA),torch.stack(newb)
            

    def preprocess(self,p,x,config):
        if p.le:
            p.le.A,p.le.b = self.scale(p.le.A,p.le.b,config.scaling_desired_scale)
        if p.eq:
            p.eq.A,p.eq.b = self.scale(p.eq.A,p.eq.b,config.scaling_desired_scale)

        if isinstance(p.obj,LinearObjective):
            p.obj.b = self.scale_m(p.obj.b,config.scaling_desired_scale)
        
        if isinstance(p.obj,QuadraticObjective):
            # logger.debug("%s,%s",p.obj.A,p.obj.b)
            n = p.obj.A.size(0)
            m = self.scale_m(torch.cat([p.obj.A,p.obj.b.unsqueeze(0)],dim=0),config.scaling_desired_scale)
            p.obj.A = m[0:n]
            p.obj.b = m[n]

        return p,x
        
    def postprocess(self,p,x,config):
        return self.origin,x
        



            