=====
Usage
=====

To use Jet20 in a project::

    import jet20



Examples
--------

* simple LP problem

.. code-block:: python

        p = Problem("test")
        x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

        p.minimize(2 * x1 + 3 * x2 + x3 + 5 * x4)

        p.constraint(x1 + x4 >= 1,
                x2 + x4 >= 1,
                x1 + x2 == 1,
                x2 + x3 == 1)

        solution = p.solve()
        print (solution.obj_value)
        print (solution.vars)
        print (solution.status)

* simple LP problem, Matrix Form

.. code-block:: python

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

        solution = p.solve(device="cuda:0",opt_tolerance=1e-3,opt_constraint_tolerance=1e-5,rouding_precision=3)
        print (solution)

* simple QP problem

.. code-block:: python

    p = Problem("test")
    x1,x2,x3,x4 = p.variables("x1,x2,x3,x4",lb=0)

    p.minimize(2*x1**2 + 3*x2**2 + x3**2 + 5*x4**2 + x1*x2 + 2*x2*x3 + 4*x1*x4)
    p.constraint(x1 + x4 >= 1,
                 x2 + x4 >= 1,
                 x1 + x2 == 1,
                 x2 + x3 == 1)

    solution = p.solve()
    print (solution)


* simple QP problem, Matrix Form

.. code-block:: python

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


    p = jet20.Problem("test")
    xs = p.variables("x1,x2,x3,x4",lb=0)
    p.minimize(jet20.quad(Q,xs) + c @ xs)
    p.constraints(A1 @ xs >= b1,
                  A2 @ xs == b2)

    solution = p.solve(device="cpu",opt_tolerance=1e-8,opt_constraint_tolerance=1e-8,rouding_precision=3)
    print (solution)
