import numpy as np
from jet20.backend import Config,Solver,EnsureEqFeasible,EnsureLeFeasible,Rounding
from jet20.backend import Problem as P
from jet20.frontend.const import *



def jet20_default_backend_func(problem,config=None,x=None):
    eps = np.finfo(np.float64).eps
    config = config or Config()

    s = Solver()
    s.register_pres(EnsureEqFeasible(),EnsureLeFeasible())
    s.register_posts(Rounding(),EnsureEqFeasible(),EnsureLeFeasible())

    var_names = [  v.name for v in problem._variables ]
    _obj, _constraints, _ops, _consts = problem.canonical

    m,n = _obj.shape
    assert m == n

    obj_Q = _obj[:n-1,:n-1]
    obj_b = _obj[n-1,:n-1] + _obj[:n-1,n-1]
    obj_c = _obj[n-1][n-1]
    obj = (obj_Q,obj_b,obj_c)

    assert _constraints.shape[0] == _ops.shape[0] == _consts.shape[0]

    eq_A = _constraints[_ops == OP_EQUAL]
    eq_b = _consts[_ops == OP_EQUAL]
    if eq_A.size > 0:
        eq = (eq_A,eq_b)
    else:
        eq = None

    _le_A = _constraints[_ops == OP_LE]
    _le_b = _consts[_ops == OP_LE]

    lt_A = _constraints[_ops == OP_LT]
    lt_b = _consts[_ops == OP_LT]
    lt_b = lt_b - eps

    le_A = np.concatenate([_le_A,lt_A])
    le_b = np.concatenate([_le_b,lt_b])

    if le_A.size > 0:
        le = (le_A,le_b)
    else:
        le = None
 
    p = P.from_numpy(var_names,obj,le,eq)
    
    return s.solve(p,config,x)


