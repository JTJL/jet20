from jet20 import Problem
import numpy as np
import pytest



def test_lp_basic():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4")

    p.minimize(2 * x1 + 3 * x2 + x3 + 5 * x4)

    p.constraint(x1 + x4 >= 1)
    p.constraint(x2 + x4 >= 1)
    p.constraint(x1 + x2 == 1)
    p.constraint(x2 + x3 == 1)

    p.constraint(x1 >= 0)
    p.constraint(x2 >= 0)
    p.constraint(x3 >= 0)
    p.constraint(x4 >= 0)

    solution = p.solve()
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



def test_lp_basic2():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

    p.minimize(2 * x1 + 3 * x2 + x3 + 5 * x4)

    p.constraint(x1 + x4 >= 1,
                x2 + x4 >= 1,
                x1 + x2 == 1,
                x2 + x3 == 1)

    solution = p.solve()
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
    p.constraint(A1 @ xs >= b1,
                A2 @ xs == b2)

    solution = p.solve()
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



def test_qp_basic1():
    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

    p.minimize(2 * x1 ** 2 + 3 * x2 * x2 + x3 ** 2 + 5 * x4 ** 2 + x1*x2 + 2*x2*x3 + 4*x1*x4)
    p.constraint(x1 + x4 >= 1,
                x2 + x4 >= 1,
                x1 + x2 == 1,
                x2 + x3 == 1)

    solution = p.solve()
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()



def test_qp_basic2():
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

    p.minimize(p.quad(Q,xs) + c @ xs + 2.0)
    p.constraint(A1 @ xs >= b1,
                A2 @ xs == b2)

    solution = p.solve()
    print (solution)

    assert solution.obj_value == 5.5
    assert (solution.x == np.array([0.5,0.5,0.5,0.5])).all()