#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['torch>=1.5.0','numpy>=1.18.2']

setup_requirements = ['pytest-runner', 'scipy']

test_requirements = ['pytest>=3', 'cvxpy','cvxopt']

setup(
    author="ioriiod0",
    author_email='ioriiod0@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="a gpu powered lp & qp solver",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='jet20',
    name='jet20',
    packages=find_packages(include=['jet20', 'jet20.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ioriiod0/jet20',
    version='0.5.0',
    zip_safe=False,
)
