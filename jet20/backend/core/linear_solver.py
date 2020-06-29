import torch



def robust_cholesky(A,beta=1e-3):
    min_aii = torch.diagonal(A).min()
    if min_aii > 0:
        t = 0
    else:
        t = -min_aii + beta
    
    I = torch.diag(A.new_ones(A.size(0)))
    while True:
        try:
            A_ = A + t * I
            l = A_.cholesky()
            return l
        except Exception as e:
            t = max(2*t,beta)


def conjugate_gradients(A, b, nsteps, M_inv = None,residual_tol=1e-3):
    x = A.new_zeros(b.size())
    r = -b.clone()
    if M_inv is not None:
        y = M_inv @ r
        p = -y.clone()
        r_2 = r @ y
    else:
        p = -r.clone()
        r_2 = r @ r 
        
    
    for i in range(nsteps):
        _Avp = A @ p
        pap = p @ _Avp
        if pap <= 0:
            if i == 0:
                return p
            else:
                return x
        
        alpha = r_2 / pap
        x += alpha * p
        r += alpha * _Avp
        if M_inv is not None:
            y = M_inv @ r
            new_r_2 = r @ y
        else:
            new_r_2 = r @ r
            
        betta = new_r_2 / r_2
        
        if M_inv is not None:
            p = -y + betta * p
        else:
            p = -r + betta * p
        r_2 = new_r_2
        
        if r_2 < residual_tol:
            break
            
    return x



class LinearSolver(object):
    def __call__(self,A,b):
        raise NotImplementedError()



class LUSolver(LinearSolver):
    def __init__(self):
        super(LUSolver,self).__init__()
        self.cache = {}

    def __call__(self,A,b):
        lup = self.cache.get(id(A))
        if lup is None:
            lup = A.lu()
            self.cache[id(A)] = lup

        if b.ndim == 1:
            return torch.lu_solve(b.unsqueeze(-1),*lup).squeeze(-1)
        else:
            return torch.lu_solve(b,*lup)


class CholeskySolver(LinearSolver):
    def __init__(self):
        super(CholeskySolver,self).__init__()
        self.cache = {}

    def __call__(self,A,b):
        l = self.cache.get(id(A))
        if l is None:
            l = robust_cholesky(A)
            self.cache[id(A)] = l

        if b.ndim == 1:
            return torch.cholesky_solve(b.unsqueeze(-1),l).squeeze(-1)
        else:
            return torch.cholesky_solve(b,l)


class CGSolver(LinearSolver):
    def __init__(self,nsteps, M_inv = None,residual_tol=1e-3):
        super(CGSolver,self).__init__()
        self.nsteps = nsteps
        self.M_inv = M_inv
        self.residual_tol = residual_tol

    def __call__(self,A,b):
        return conjugate_gradients(A,b,self.nsteps,self.M_inv,self.residual_tol)