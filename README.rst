============
|logo| Jet20
============

.. |logo| image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200707111702.png
        :width: 80
        :height: 80


.. image:: https://img.shields.io/pypi/v/jet20.svg
        :target: https://pypi.python.org/pypi/jet20

.. image:: https://img.shields.io/travis/ioriiod0/jet20.svg
        :target: https://travis-ci.com/ioriiod0/jet20

.. image:: https://readthedocs.org/projects/jet20/badge/?version=latest
        :target: https://jet20.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Jet20 is a GPU Powered LP & QP solver. It provides three main features:

- A frontend for modeling Lp & QP problem easily
- A backend written by pytorch
- Modular design, easy to extend

.. note::

        * This is an alpha release and the project is still under heavy development. 
        * CrossOver is not supported currently.
        * There is still rooms for improvement of memory usage, computation performance and stability.
        * PRs and issues are welcome.


Performance
-----------

Benchmark on random generated LP problems with different sizes and density (ratio of non-zeros).

        * Cplex running on Intel i9/2.3GHz/8cores while Jet20 running on single Nvidia 2080Ti.
        * The size of the problem equals N refers to N varibles and N constraints.

.. image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200703193107.png
        :width: 300
        :height: 200
.. image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200703193213.png
        :width: 300
        :height: 200
.. image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200703193153.png
        :width: 300
        :height: 200
.. image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200703193140.png
        :width: 300
        :height: 200
.. image:: http://markdown-1254266567.cosgz.myqcloud.com/img/20200703193124.png
        :width: 300
        :height: 200


Installation
------------

To install Jet20, run this command in the terminal:

.. code-block:: console

    $ pip install jet20

This is the preferred method to install Jet20, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can walk
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

* Free software: MIT license
* Documentation: https://jet20.readthedocs.io.


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



Road Map
--------

- [ ] sparse tensor support
- [ ] crossover support
- [ ] stability improvement
- [ ] more features

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
