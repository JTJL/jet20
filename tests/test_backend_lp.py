#!/usr/bin/env python

"""Tests for `jet20` package."""

import pytest
import torch

torch.set_printoptions(precision=10)
import numpy as np
from cvxopt.base import matrix, spmatrix
from cvxopt.modeling import op
import os


import logging
logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

from jet20.backend import (Solver,EnsureEqFeasible,EnsureLeFeasible,
                            Rounding,Config,Problem,LinearEqConstraints,
                            LinearLeConstraints,LinearObjective,
                            EqConstraitConflict,LeConstraitConflict)


def read_mps_preprocess(filepath):
    os.system("./emps %s >> tmp.mps" % filepath)
    problem = op()
    problem.fromfile('tmp.mps')
    mat_form = problem._inmatrixform(format='dense')
    format = 'dense'
    assert mat_form
    lp, vmap, mmap = mat_form

    variables = lp.variables()
    x = variables[0]
    c = lp.objective._linear._coeff[x]
    inequalities = lp._inequalities
    G = inequalities[0]._f._linear._coeff[x]
    h = -inequalities[0]._f._constant
    equalities = lp._equalities
    A, b = None, None
    if equalities:
        A = equalities[0]._f._linear._coeff[x]
        b = -equalities[0]._f._constant
    elif format == 'dense':
        A = matrix(0.0, (0,len(x)))
        b = matrix(0.0, (0,1))
    else:
        A = spmatrix(0.0, [], [],  (0,len(x)))  # CRITICAL
        b = matrix(0.0, (0,1))

    c = np.array(c).flatten()
    G = np.array(G)
    h = np.array(h).flatten()
    A = np.array(A)
    b = np.array(b).flatten()

    return c, G, h, A, b


@pytest.fixture
def basic_solver():
    s = Solver()
    # simpify = Simpify()
    s.register_pres(EnsureEqFeasible(),EnsureLeFeasible())
    s.register_posts(Rounding(),EnsureEqFeasible(),EnsureLeFeasible())
    return s


@pytest.fixture
def benchmark_problem():
    c,G,h,A,b = read_mps_preprocess("cre-a")
    _vars = [ "x_%s" % i for i in range(c.size) ]
    return Problem.from_numpy(_vars,(None,c,None),(G,h),(A,b))


@pytest.fixture
def easy_lp_problem():
    scale = 1

    LE_A = -1 * np.array([
        [1,0,0,1], # >= 1
        [0,1,0,1], # >= 1
        [1,0,0,0], 
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]) * scale

    LE_B = -1 * np.array([1,1,0,0,0,0]) * scale

    EQ_A = np.array([
        [1,1,0,0], #==1
        [0,1,1,0], #==1
    ]) * scale

    EQ_B = np.array([1,1]) * scale

    OBJ_C = np.array([2,3,1,5]) * scale

    _vars = [ "x_%s" % i for i in range(4) ]
    return Problem.from_numpy(_vars,(None,OBJ_C,None),(LE_A,LE_B),(EQ_A,EQ_B),torch.device("cpu"),torch.float32)



@pytest.fixture
def bad_eq_lp_problem():
    LE_A = -1 * torch.FloatTensor([
        [1,0,0,1], # >= 1
        [0,1,0,1], # >= 1
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])

    LE_B = -1 * torch.FloatTensor([1,1,0,0,0,0])

    EQ_A = torch.FloatTensor([
        [1,1,0,0], #==1
        [1,1,0,0], #==2
    ])

    EQ_B = torch.FloatTensor([1,2])

    OBJ_C = torch.FloatTensor([2,3,1,5])

    eq = LinearEqConstraints(EQ_A,EQ_B)
    le = LinearLeConstraints(LE_A,LE_B)
    obj = LinearObjective(OBJ_C)

    _vars = [ "x_%s" % i for i in range(4) ]
    return Problem(_vars,obj,le,eq)


@pytest.fixture
def bad_le_lp_problem():
    LE_A = -1 *torch.FloatTensor([
        [-1,-1,0,-1], # <= 0.5
        [1,0,0,1], # >= 1
        [0,1,0,1], # >= 1
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])

    LE_B = -1 * torch.FloatTensor([-0.5,1,1,0,0,0,0])

    EQ_A = torch.FloatTensor([
        [1,1,0,0], #==1
        [0,1,1,0], #==2
    ])

    EQ_B = torch.FloatTensor([1,1])

    OBJ_C = torch.FloatTensor([2,3,1,5])
    _vars = [ "x_%s" % i for i in range(4) ]

    return Problem.from_numpy(_vars,(None,OBJ_C,None),(LE_A,LE_B),(EQ_A,EQ_B),torch.device("cpu"),torch.float32)


@pytest.fixture
def random_benchmark_problem():
    import numpy as np
    np.random.seed(42)

    N = 400
    M1 = 800
    R = 40

    A1 = np.zeros((M1,N))

    for i in range(M1):
        indexs = np.random.choice(N, size=R, replace=False)
        A1[i,indexs] = np.abs(np.random.randn(R))

    A1 = -1 * A1
    A2 = -1 * np.diag(np.ones(N))

    A = np.concatenate([A1,A2],axis=0)
        
    B1 = -1 * np.ones(M1)
    B2 = -1 * np.zeros(N)

    B = np.concatenate([B1,B2])

    b = np.ones(N)

    _vars = [ "x_%s" % i for i in range(N) ]

    p = Problem.from_numpy(_vars,(None,b,None),(A,B),None,torch.device("cpu"),torch.float32)    
    return p


def test_basic(basic_solver,easy_lp_problem):
    solution = basic_solver.solve(easy_lp_problem,Config(device="cpu",opt_tolerance=1e-5))
    print (solution)
    print (solution.x)
    # assert solution.obj_value == 55
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()


# def test_bad_lp_problem(basic_solver,bad_eq_lp_problem):
#     with pytest.raises(EqConstraitConflict):
#         basic_solver.solve(bad_eq_lp_problem,Config())


def test_bad_eq_problem(basic_solver,bad_le_lp_problem):
    with pytest.raises(LeConstraitConflict):
        basic_solver.solve(bad_le_lp_problem,Config())


def test_random_benchmark(basic_solver,random_benchmark_problem):
    solution = basic_solver.solve(random_benchmark_problem,Config(device="cpu",opt_tolerance=1e-8))
    print (solution)
    13.470434285878028
    assert solution.obj_value <= 13.470434285878028


def test_benchmark(basic_solver,benchmark_problem):
    solution = basic_solver.solve(benchmark_problem,Config(device="cpu",opt_tolerance=1e-8))
    print (solution)
    assert solution.obj_value <= 5501.846005

