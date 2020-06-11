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
                    EnsureLeFeasible,Rounding,Config,Problem,
                    LinearEqConstraints,LinearLeConstraints,
                    QuadraticObjective,EqConstraitConflict,LeConstraitConflict)


@pytest.fixture
def solver():
    s = Solver()
    s.register_pres(EnsureEqFeasible(),EnsureLeFeasible())
    s.register_posts(Rounding())
    return s
    
    
@pytest.fixture
def easy_qp_problem():
    LE_A = -1 * torch.FloatTensor([[1,1,0,0],
                               [0,0,1,1]])
    LE_B = -1 * torch.FloatTensor([1,1])

    # EQ_A = -1 * torch.FloatTensor([[0,0,0,0],
    #                            [0,0,1,1]])
    # EQ_B = -1 * torch.FloatTensor([1,1])


    OBJ_A = torch.FloatTensor(np.diag([1,1,1,1]))
    OBJ_C = torch.FloatTensor([1,1,1,1])

    eq = None
    # eq = LinearEqConstraints(EQ_A,EQ_B)
    le = LinearLeConstraints(LE_A,LE_B)
    obj = QuadraticObjective(OBJ_A,OBJ_C)

    return Problem(obj,le,eq)


def test_basic(solver,easy_qp_problem):
    solution = solver.solve(easy_qp_problem,Config())
    print (solution)
    # assert solution.obj_value == 5.5
    # assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



