#!/usr/bin/env python

"""Tests for `jet20` package."""

import pytest
import torch
import numpy as np
torch.set_printoptions(precision=10)

import logging
logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

from jet20.backend import (Solver,EnsureEqFeasible,EnsureLeFeasible,
                            Scaling,Rounding,Config,Problem,LinearEqConstraints,
                            LinearLeConstraints,LinearObjective,
                            EqConstraitConflict,LeConstraitConflict,
                            Simpify)

@pytest.fixture
def solver():
    s = Solver()
    simpify = Simpify()
    s.register_pres(simpify,Scaling(),EnsureLeFeasible())
    s.register_posts(simpify,Rounding(),EnsureEqFeasible(),EnsureLeFeasible())
    return s
    
    
##max_scale : 10000000
##min_scale : 

@pytest.fixture
def easy_lp_problem():
    scale = 0.00001

    LE_A = -1 *torch.DoubleTensor([
        [1,0,0,1], # >= 1
        [0,1,0,1], # >= 1
        [1,0,0,0], 
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]) * scale

    LE_B = -1 * torch.DoubleTensor([1,1,0,0,0,0]) * scale

    EQ_A = torch.DoubleTensor([
        [1,1,0,0], #==1
        [0,1,1,0], #==1
    ]) * scale

    EQ_B = torch.DoubleTensor([1,1]) * scale

    OBJ_C = torch.DoubleTensor([2,3,1,5]) * scale

    eq = LinearEqConstraints(EQ_A,EQ_B)
    le = LinearLeConstraints(LE_A,LE_B)
    obj = LinearObjective(OBJ_C)

    _vars = [ "x_%s" for i in range(4) ]
    return Problem(_vars,obj,le,eq)


@pytest.fixture
def easy_lp_problem2():

    A = np.load("A.npy")
    B = np.load("B.npy")
    C = np.ones(A.shape[1])

    A = torch.DoubleTensor(A)
    B = torch.DoubleTensor(B)
    OBJ_C = torch.DoubleTensor(C)

    eq = None
    le = LinearLeConstraints(A,B)
    obj = LinearObjective(OBJ_C)

    _vars = [ "x_%s" for i in range(A.shape[1]) ]
    return Problem(_vars,obj,le,eq)

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

    _vars = [ "x_%s" for i in range(4) ]
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

    eq = LinearEqConstraints(EQ_A,EQ_B)
    le = LinearLeConstraints(LE_A,LE_B)
    obj = LinearObjective(OBJ_C)

    _vars = [ "x_%s" for i in range(4) ]
    return Problem(_vars,obj,le,eq)



def test_basic(solver,easy_lp_problem):
    solution = solver.solve(easy_lp_problem,Config(opt_max_cnt=100,opt_tolerance=1e-20))
    print (solution)
    # assert solution.obj_value == 55
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()


def test_bad_lp_problem(solver,bad_eq_lp_problem):
    with pytest.raises(EqConstraitConflict):
        solver.solve(bad_eq_lp_problem,Config())


def test_bad_eq_problem(solver,bad_le_lp_problem):
    with pytest.raises(LeConstraitConflict):
        solver.solve(bad_le_lp_problem,Config())


def test_basic_eq_problem_2(solver,easy_lp_problem2):
    solution = solver.solve(easy_lp_problem2,Config(opt_max_cnt=100,opt_tolerance=1e-20))
    print (solution)
    assert solution.obj_value <= 9.983
    # assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()


# def test_scale(easy_lp_problem):
#     print ("before.......")
#     print (easy_lp_problem.le.A)
#     print (easy_lp_problem.le.b)
#     print (easy_lp_problem.obj.c)

#     scale = Scaling()
#     easy_lp_problem,_ = scale(easy_lp_problem,None,Config())

#     print ("after.......")
#     print (easy_lp_problem.le.A)
#     print (easy_lp_problem.le.b)
#     print (easy_lp_problem.obj.c)

