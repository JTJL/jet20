
import torch

    
class LinearObjective(object):
    def __init__(self,c):
        super(LinearObjective,self).__init__()
        self.c = c
        
    def __call__(self,x):
        return torch.dot(self.c,x)


class QuadraticObjective(object):
    def __init__(self,A,c):
        super(QuadraticObjective,self).__init__()
        self.A = A
        self.c = c
        
    def __call__(self,x):
        return torch.dot(x,torch.mv(self.A,x)) + torch.dot(self.c,x)