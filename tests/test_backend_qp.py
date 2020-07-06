#!/usr/bin/env python

"""Tests for `jet20` package."""

import pytest
import torch
import numpy as np
torch.set_printoptions(precision=10)

import logging
logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

from jet20.backend import (Solver,EnsureEqFeasible,
                    EnsureLeFeasible,Rounding,Config,Problem,QUADRATIC,LINEAR,
                    LinearEqConstraints,LinearLeConstraints,LambdaObjective,
                    QuadraticObjective,EqConstraitConflict,LeConstraitConflict)



@pytest.fixture
def basic_solver():
    s = Solver()
    # simpify = Simpify()
    s.register_pres(EnsureEqFeasible(),EnsureLeFeasible())
    s.register_posts(Rounding(),EnsureEqFeasible(),EnsureLeFeasible())
    return s
    


@pytest.fixture
def random_eq_qp_problem():
    import numpy as np
    import scipy.sparse as spa
    import cvxpy

    np.random.seed(42)
    n = 100

    m = int(n/2)

    # Generate problem data
    n = int(n)
    m = m
    P = spa.random(n, n, density=0.15,
                    data_rvs=np.random.randn,
                    format='csc')
    P = P.dot(P.T).tocsc() + 1e-02 * spa.eye(n)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15,
                        data_rvs=np.random.randn,
                        format='csc')
    x_sol = np.random.randn(n)  # Create fictitious solution
    l = A@x_sol
    u = np.copy(l)

    _vars = [ "x_%s" % i for i in range(n) ]

    p = Problem.from_numpy(_vars,(0.5 * P.todense(),q,None),None,(A.todense(),u),torch.device("cpu"),torch.float64)
    return p


@pytest.fixture
def easy_qp_problem():
    LE_A = -1 * torch.DoubleTensor([[1,1,0,0],
                               [0,0,1,1]])
    LE_B = -1 * torch.DoubleTensor([1,1])

    # EQ_A = -1 * torch.DoubleTensor([[0,0,0,0],
    #                            [0,0,1,1]])
    # EQ_B = -1 * torch.DoubleTensor([1,1])


    OBJ_A = torch.DoubleTensor(np.diag([1,1,1,1]))
    OBJ_B = torch.DoubleTensor([1,1,1,1])

    eq = None
    # eq = LinearEqConstraints(EQ_A,EQ_B)
    le = LinearLeConstraints(LE_A,LE_B)
    obj = QuadraticObjective(OBJ_A,OBJ_B)

    _vars = [ "x_%s" % i for i in range(4) ]
    x = torch.ones(4).double()
    print ("obj:",obj(x))

    return Problem(_vars,obj,le,eq)



@pytest.fixture
def random_le_qp_problem():
    import numpy as np
    import scipy.sparse as spa
    import cvxpy

    np.random.seed(42)

    n = 100
    m = int(n * 2)


    P = spa.random(n, n, density=0.15,
                    data_rvs=np.random.randn,
                    format='csc')
    P = P.dot(P.T).tocsc() + 1e-02 * spa.eye(n)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15,
                        data_rvs=np.random.randn,
                        format='csc')
    v = np.random.randn(n)   # Fictitious solution
    delta = np.abs(np.random.rand(m))  # To get inequality
    u = A@v + delta
    # l = - np.inf * np.ones(m)  # u - np.random.rand(m)
    _vars = [ "x_%s" % i for i in range(n) ]

    p = Problem.from_numpy(_vars,(0.5 * P.todense(),q,None),(A.todense(),u),None,torch.device("cpu"),torch.float64)
    return p


def test_basic(basic_solver,easy_qp_problem):
    solution = basic_solver.solve(easy_qp_problem,Config(device="cpu",))
    print (solution)
    # assert solution.obj_value == 5.5
    # assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



def test_random_eq_qp_problem(basic_solver,random_eq_qp_problem):
    solution = basic_solver.solve(random_eq_qp_problem,Config(device="cpu",opt_tolerance=1e-5,opt_constraint_tolerance=1e-5))
    print (solution)
    print (random_eq_qp_problem.eq(torch.tensor(solution.x)))
    assert random_eq_qp_problem.eq.validate(torch.tensor(solution.x),1e-5)
    assert solution.obj_value < 151.36997


def test_random_le_qp_problem(basic_solver,random_le_qp_problem):
    solution = basic_solver.solve(random_le_qp_problem,Config(device="cpu",opt_tolerance=1e-8,opt_constraint_tolerance=1e-5))
    print (solution)
    assert random_le_qp_problem.le.validate(torch.tensor(solution.x))
    assert solution.obj_value < 396.798781