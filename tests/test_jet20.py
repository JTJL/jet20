from jet20 import Problem
import numpy as np
import pytest



def test_basic():
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


