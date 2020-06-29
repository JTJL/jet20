
import torch
from jet20.backend.const import LINEAR,QUADRATIC

import logging
logger = logging.getLogger(__name__)


class Objective(object):

    def __call__(self,x):
        raise NotImplementedError("")

    def type(self):
        raise NotImplementedError("")

    def float(self):
        raise NotImplementedError("")

    def double(self):
        raise NotImplementedError("")

    def to(self,device):
        raise NotImplementedError("")


class LambdaObjective(Objective):
    
    def __init__(self,_type,f,*args):
        super(LambdaObjective,self).__init__()
        self._type = _type
        self.f = f
        self.args = args

    def __call__(self,x):
        return self.f(x,*self.args)
        
        
    def type(self):
        return self._type

        
    def float(self):
        self.args = [ arg.float() for arg in self.args ]


    def double(self):
        self.args = [ arg.double() for arg in self.args ]


    def to(self,device):
        self.args = [ arg.to(device) for arg in self.args ]


class LinearObjective(Objective):
    def __init__(self,b,c=None):
        super(LinearObjective,self).__init__()
        self.b = b
        self.c = c if c is not None else 0.0

    def __call__(self,x):
        return self.b @ x + self.c

    def type(self):
        return LINEAR

    def float(self):
        self.b = self.b.float()
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.float()


    def double(self):
        self.b = self.b.double()
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.double()


    def to(self,device):
        self.b = self.b.to(device)
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.to(device)
        

class QuadraticObjective(Objective):
    def __init__(self,Q,b=None,c=None):
        super(QuadraticObjective,self).__init__()
        self.Q = Q
        self.b = b if b is not None else Q.new_zeros(Q.size(0))
        self.c = c if c is not None else 0.0

    def __call__(self,x):
        return x @ self.Q @ x + self.b @ x + self.c

    def type(self):
        return QUADRATIC

    def float(self):
        self.Q = self.Q.float()
        self.b = self.b.float()
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.float()


    def double(self):
        self.Q = self.Q.double()
        self.b = self.b.double()
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.double()


    def to(self,device):
        self.Q = self.Q.to(device)
        self.b = self.b.to(device)
        if isinstance(self.c,torch.Tensor):
            self.c = self.c.to(device)