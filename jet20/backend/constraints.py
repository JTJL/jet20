

import torch



class LeConstraitConflict(Exception):
    pass

class EqConstraitConflict(Exception):
    pass


class LinearEqConstraints(object):
    def __init__(self,A,b):
        super(LinearEqConstraints,self).__init__()
        self.A = A
        self.b = b
        
    def __call__(self,x):
        return torch.mv(self.A,x) - self.b
    
    def validate(self,x,tolerance=0.0):
        x = torch.abs(self(x))
        neq = x > tolerance
        return neq.float().sum() == 0


class LinearLeConstraints(object):
    def __init__(self,A,b):
        super(LinearLeConstraints,self).__init__()
        self.A = A
        self.b = b
        
    def __call__(self,x):
        return torch.mv(self.A,x) - self.b
    
    def validate(self,x):
        nle = self(x) > 0
        return nle.float().sum() == 0

