

import torch

from jet20.backend.const import LINEAR,QUADRATIC


class LeConstraitConflict(Exception):
    pass

class EqConstraitConflict(Exception):
    pass


class Constraints(object):
    def __call__(self,x):
        raise NotImplementedError("")
        
    def validate(self,x,*args,**kwargs):
        raise NotImplementedError("")
        
    def type(self):
        raise NotImplementedError("")
        
    def size(self):
        raise NotImplementedError("")

        
    def float(self):
        raise NotImplementedError("")


    def double(self):
        raise NotImplementedError("")


    def to(self,device):
        raise NotImplementedError("")


class LinearConstraints(Constraints):
    def __init__(self,A,b):
        super(LinearConstraints,self).__init__()
        self.A = A
        self.b = b

    def __call__(self,x):
        return self.A @ x - self.b
        
    def validate(self,x,*args,**kwargs):
        raise NotImplementedError("")
        
    def type(self):
        return LINEAR
        
    def size(self):
        return self.A.size(0)

    def float(self):
        A = self.A.float()
        b = self.b.float()
        return self.__class__(A,b)


    def double(self):
        A = self.A.double()
        b = self.b.double()
        return self.__class__(A,b)


    def to(self,device):
        A = self.A.to(device)
        b = self.b.to(device)
        return self.__class__(A,b)


class LinearEqConstraints(LinearConstraints):
    def __init__(self,A,b):
        super(LinearEqConstraints,self).__init__(A,b)
        
    def validate(self,x,tolerance=1e-8):
        x = torch.abs(self(x))
        neq = x > tolerance
        return neq.float().sum() == 0


class LinearLeConstraints(LinearConstraints):
    def __init__(self,A,b):
        super(LinearLeConstraints,self).__init__(A,b)

    def validate(self,x):
        nle = self(x) > 0
        return nle.float().sum() == 0

