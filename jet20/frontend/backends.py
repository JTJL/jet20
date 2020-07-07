import numpy as np
import torch

from jet20.backend import Config,Solver,EnsureEqFeasible,EnsureLeFeasible,Rounding,Solution,SUB_OPTIMAL,OPTIMAL,FAIELD,USER_STOPPED
from jet20.backend import Problem as P
from jet20.frontend.const import *



def jet20_default_backend_func(problem,x=None,opt_tolerance=1e-3,
                                opt_u = 10.0,
                                opt_alpha = 0.1,
                                opt_beta = 0.5,
                                opt_constraint_tolerance = 1e-5,
                                opt_verbose = False,
                                rouding_precision = 3,
                                force_rouding = False,
                                device ="cuda"):

    """
    This function is a wrapper of jet20 backend.

    :param problem: the problem instance.
    :type problem: class:`jet20.Problem`.
    :param x: initial solution of the problem
    :type x: list,numpy.ndarray
    :param opt_u: hyperparameters for interior point method
    :type opt_u: float
    :param opt_alpha: hyperparameters for line search
    :type opt_alpha: float
    :param opt_beta: hyperparameters for line search
    :type opt_beta: float
    :param opt_tolerance: objective value tolerance
    :type opt_tolerance: float
    :param opt_constraint_tolerance: feasibility tolerance
    :type opt_constraint_tolerance: float
    :param rouding_precision: rouding precision
    :type rouding_precision: int
    :param force_rouding: whether force rounding
    :type rouding_precision: bool
    :return: solution of the problem
    :rtype: Solution
    """

    eps = np.finfo(np.float64).eps
    config = Config(opt_tolerance=opt_tolerance,opt_u=opt_u,opt_alpha=opt_alpha,
                opt_beta=opt_beta,opt_constraint_tolerance=opt_constraint_tolerance,
                opt_verbose=opt_verbose,rouding_precision=rouding_precision,
                force_rouding=force_rouding,device=device)

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
 
    p = P.from_numpy(var_names,obj,le,eq,dtype=torch.float64,device=config.device)
    if x is not None:
        x = torch.tensor(x,dtype=torch.float64,device=config.device)
    
    return s.solve(p,config,x)


