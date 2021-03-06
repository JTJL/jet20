from jet20 import Problem,quad
import numpy as np
import pytest



def test_lp_basic():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4")

    p.minimize(2 * x1 + 3 * x2 + x3 + 5 * x4)

    p.constraints(x1 + x4 >= 1)
    p.constraints(x2 + x4 >= 1)
    p.constraints(x1 + x2 == 1)
    p.constraints(x2 + x3 == 1)

    p.constraints(x1 >= 0)
    p.constraints(x2 >= 0)
    p.constraints(x3 >= 0)
    p.constraints(x4 >= 0)

    print ("p.canonical:",p.canonical)

    solution = p.solve(device="cpu")
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



def test_lp_basic2():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

    p.minimize(2 * x1 + 3 * x2 + x3 + 5 * x4)

    p.constraints(x1 + x4 >= 1,
                x2 + x4 >= 1,
                x1 + x2 == 1,
                x2 + x3 == 1)

    print ("p.canonical:",p.canonical)
    solution = p.solve(device="cpu")
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()


def test_lp_basic3():
    A1 = np.array([ [1,0,0,1],
                    [0,1,0,1]  ])
    b1 = 1

    A2 = np.array([ [1,1,0,0],
                    [0,1,1,0]  ])
    b2 = np.array([1,1])

    c = np.array([2,3,1,5])


    p = Problem("test")
    xs = p.variables("x1,x2,x3,x4",lb=0)
    
    p.minimize(c @ xs)
    p.constraints(A1 @ xs >= b1,
                A2 @ xs == b2)

    solution = p.solve(device="cpu")
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()


def test_qp_basic1():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

    p.minimize(2*x1**2 + 3*x2**2 + x3**2 + 5*x4**2 + x1*x2 + 2*x2*x3 + 4*x1*x4)
    p.constraints(x1 + x4 >= 1,
                x2 + x4 >= 1,
                x1 + x2 == 1,
                x2 + x3 == 1)

    solution = p.solve(device="cpu")
    print (solution)
    assert solution.obj_value == 4.5



def test_qp_basic2():
    np.random.seed(42)
    A1 = np.array([ [1,0,0,1],
                    [0,1,0,1]  ])
    b1 = 1
    A2 = np.array([ [1,1,0,0],
                    [0,1,1,0]  ])
    b2 = np.array([1,1])
    c = np.array([2,3,1,5])
    Q = np.random.randn(4,4)
    Q = Q.T @ Q


    p = Problem("test")
    xs = p.variables("x1,x2,x3,x4",lb=0)
    p.minimize(quad(Q,xs) + c @ xs)
    p.constraints(A1 @ xs >= b1,
                A2 @ xs == b2)

    solution = p.solve(device="cpu",opt_tolerance=1e-8,opt_constraint_tolerance=1e-8,rouding_precision=3)
    print (solution)
    print (solution.obj_value)
    print (solution.vars)
    print (solution.status)

    assert solution.obj_value <= 12.081916655

