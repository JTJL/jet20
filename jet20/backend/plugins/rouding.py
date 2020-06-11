import torch

from jet20.backend.plugins import Plugin

import logging
logger = logging.getLogger(__name__)


def round_(x,n=3):
    return (x * 10**n).round() / (10**n)



class Rounding(Plugin):

    def postprocess(self,p,x,config):
        old_value = self.p.obj(x)

        for i in range(config.rouding_precision,16):
            _x = round_(x,i)

            if p.eq and not p.eq.validate(x,config.eq_constraint_tolerance):
                continue

            if p.le and not p.le.validate(x):
                continue
            
            new_value = p.obj(_x) 
            if new_value > old_value:
                logger.warn("objective get worse,before rouding: %s, after rouding:%s",old_value.item(),new_value.item())

            return p,x
        
        logger.warn("rouding faild.")
        return p,x

            
            

