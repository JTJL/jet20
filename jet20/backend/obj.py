
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
        args = [ arg.float() for arg in self.args ]
        return self.__class__(self._type,self.f,*args)

    def double(self):
        args = [ arg.double() for arg in self.args ]
        return self.__class__(self._type,self.f,*args)


    def to(self,device):
        args = [ arg.to(device) for arg in self.args ]
        return self.__class__(self._type,self.f,*args)


class LinearObjective(Objective):
    def __init__(self,b,c=None):
        super(LinearObjective,self).__init__()
        self.b = b
        c = c if c is not None else 0.0
        self.c = torch.tensor(c,dtype=b.dtype,device=b.device)

    def __call__(self,x):
        return self.b @ x + self.c

    def type(self):
        return LINEAR

    def float(self):
        b = self.b.float()
        c = self.c.float()

        return self.__class__(b,c)


    def double(self):
        b = self.b.double()
        c = self.c.double()

        return self.__class__(b,c)


    def to(self,device):
        b = self.b.to(device)
        c = self.c.to(device)

        return self.__class__(b,c)
        

class QuadraticObjective(Objective):
    def __init__(self,Q,b=None,c=None):
        super(QuadraticObjective,self).__init__()
        self.Q = Q
        self.b = b if b is not None else Q.new_zeros(Q.size(0))

        c = c if c is not None else 0.0
        self.c = torch.tensor(c,dtype=Q.dtype,device=Q.device)

    def __call__(self,x):
        return x @ self.Q @ x + self.b @ x + self.c

    def type(self):
        return QUADRATIC

    def float(self):
        Q = self.Q.float()
        b = self.b.float()
        c = self.c.float()

        return self.__class__(Q,b,c)


    def double(self):
        Q = self.Q.double()
        b = self.b.double()
        c = self.c.double()

        return self.__class__(Q,b,c)


    def to(self,device):
        Q = self.Q.to(device)
        b = self.b.to(device)
        c = self.c.to(device)

        return self.__class__(Q,b,c)