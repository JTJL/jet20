============
|logo| Jet20
============

.. |logo| image:: imgs/jet20.png
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
- A GPU powered backend which impliments primal dual interior point method, written by pytorch
- Modular design, easy to extend

.. note::

        * This is an alpha release and this project is still under heavily development. 
        * CrossOver is not supported currently.
        * There still are many rooms for memory usage and performance.
        * PRs and issues are welcome.


Performance
-----------

Benchmark on random generated LP problems with different size and density(ration of non-zeros).

        * Cplex running on Intel i9/2.3GHz/8cores while Jet20 running on single Nvidia 2080Ti.
        * Problem size N means problem with N varibles and N constraits.

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

Examples
--------

* TODO

Road Map
--------

- [ ] sparse tensor support
- [ ] crossover support
- [ ] preprocessing ..

Install
--------

To install Jet20, run this command in your terminal:

.. code-block:: console

    $ pip install jet20

This is the preferred method to install Jet20, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

* Free software: MIT license
* Documentation: https://jet20.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
