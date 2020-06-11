
import torch

import logging
logger = logging.getLogger(__name__)

class LinearObjective(object):
    def __init__(self,b,c=0):
        super(LinearObjective,self).__init__()
        self.b = b
        self.c = c
        
    def __call__(self,x):
        return torch.dot(self.b,x) + self.c


class QuadraticObjective(object):
    def __init__(self,A,b=0,c=0):
        super(QuadraticObjective,self).__init__()
        self.A = A
        self.b = b
        self.c = c
        
    def __call__(self,x):
        return torch.dot(x,torch.mv(self.A,x)) + torch.dot(self.b,x) + self.c