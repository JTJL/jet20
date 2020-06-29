from typing import TypeVar, Union
from jet20.frontend import Expression
import numpy as np
import pytest

_E = TypeVar("_E", bound=BaseException)


def assert_expression(expr: Expression, expect: tuple):
    assert (expr.core_mat == expect[0]).all()
    assert (expr.linear_complete_vector == expect[1]).all()
    assert (expr.highest_order, expr.shape,
            expr.const,
            expr.is_constraint,
            expr.dim, expr.total_variables, expr.op) == expect[2:]


# expect: (core_mat,linear_complete_vector,highest_order,shape,const,is_constraint,dim,total_variables,op)
@pytest.mark.parametrize("input_data, expect, expect_exception, except_exception_match",
                         [
                             (  # expression is a single constant
                                 (np.array([[1]]),),
                                 (np.array([[1]]), np.array([1]), 0, (1, 1), 1, False, 1, 0, ''),
                                 None,
                                 ""
                             ),
                             (  # invalid shape
                                 (np.array([1]),),
                                 (),
                                 TypeError,
                                 "mat must be a 2 dimension matrix"
                             ),
                             (  # invalid shape
                                 (np.array(1),),
                                 (),
                                 TypeError,
                                 "mat must be a 2 dimension matrix"
                             ),
                             (  # (x+y)**2
                                 (np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),),
                                 (np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]), np.zeros((3, 1)), 2, (3, 3), 0, False, 3,
                                  2, ''),
                                 None,
                                 ""
                             ),
                             (  # (x+y+4)
                                 (np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, 4]]),),
                                 (
                                     np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, 4]]), np.array([1, 1, 4]), 1,
                                     (3, 3), 4,
                                     False, 3,
                                     2, ''),
                                 None,
                                 ""
                             ),
                             (  # (1*x0+2*x1+3*x2+4*x3+100)
                                 (np.array([[0, 0, 0, 0, 0.5], [0, 0, 0, 0, 1.], [0, 0, 0, 0, 1.5], [0, 0, 0, 0, 2.],
                                            [0.5, 1., 1.5, 2., 100.]]),),
                                 (np.array([[0, 0, 0, 0, 0.5], [0, 0, 0, 0, 1.], [0, 0, 0, 0, 1.5], [0, 0, 0, 0, 2.],
                                            [0.5, 1., 1.5, 2., 100.]]), np.array([1, 2, 3, 4, 100]), 1, (5, 5), 100,
                                  False, 5,
                                  4, ''),
                                 None,
                                 ""
                             ),
                             (  # x+y < 2
                                 (np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, -2]]), '<'),
                                 (
                                     np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, -2]]), np.array([1, 1, -2]), 1,
                                     (3, 3), -2,
                                     True, 3,
                                     2, '<'),
                                 None,
                                 ""
                             ),
                             (  # x+y >= 2
                                 (np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, -2]]), '>='),
                                 (
                                     np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, -2]]), np.array([1, 1, -2]), 1,
                                     (3, 3), -2,
                                     True, 3,
                                     2, '>='),
                                 None,
                                 ""
                             ),
                             (  # 4.4*x, in index, para way
                                 (None, '', 0, 4.4),
                                 (
                                     np.array([[0, 2.2], [2.2, 0]]), np.array([4.4, 0]), 1,
                                     (2, 2), 0,
                                     False, 2,
                                     1, ''),
                                 None,
                                 ""
                             ),
                             (  # -3*z < 0, in index, para way
                                 (None, '<', 2, -3),
                                 (
                                     np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1.5], [0, 0, -1.5, 0]]),
                                     np.array([0, 0, -3, 0]), 1,
                                     (4, 4), 0,
                                     True, 4,
                                     3, '<'),
                                 None,
                                 ""
                             )
                         ])
def test_init(input_data: tuple, expect: tuple, expect_exception: _E, except_exception_match: str):
    if expect_exception is not None:
        with pytest.raises(expect_exception, match=except_exception_match):
            expr = Expression(*input_data)
            assert_expression(expr, expect)
    else:
        expr = Expression(*input_data)
        assert_expression(expr, expect)


# input_data: (expr, expand to n)
@pytest.mark.parametrize("input_data, expect, expect_exception, expect_exception_match", [
    ((Expression(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])), 5),
     Expression(np.array([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 0, 1]])),
     None, ""),  # 4*4 => 5*5
    ((Expression(np.array([[1, 1], [1, 1]])), 6),
     Expression(np.array(
         [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 1]])),
     None, ""),  # 2*2 => 6*6
    ((Expression(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])), 4), None, ValueError,
     "expand dim less than the current is not allowed"),
    ((Expression(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])), 3), None, ValueError,
     "expand dim less than the current is not allowed")
    # ((Expression(np.array([[]])), 3), Expression()),
])
def test_expand(input_data: tuple, expect: Expression, expect_exception: _E, expect_exception_match: str):
    if expect_exception is not None:
        with pytest.raises(expect_exception, match=expect_exception_match):
            assert input_data[0].expand(input_data[1]).equal(expect)
    else:
        print(input_data[0].expand(input_data[1]), expect, sep='\n')
        assert input_data[0].expand(input_data[1]).equal(expect)


@pytest.mark.parametrize("input_data, expect, expect_exception, expect_exception_match", [
    (  # x**2 + y**2 + z**2 + 2xy + 2xz + 2yz + 2x + 2y + 2z + 1
        Expression(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])),
        np.array([2, 2, 2, 1]),  # 2x + 2y + 2z + 1
        None, ""
    ),
    (  # x**2 + y**2 + z**2 + 3xy + 5xz + 2yz + 3.2x + 3.4y - 3.6z - 1.9
        Expression(np.array([[1, 1.5, 2.5, 1.6], [1.5, 1, 1, 1.7], [2.5, 1, 1, -1.8], [1.6, 1.7, -1.8, -1.9]])),
        np.array([3.2, 3.4, -3.6, -1.9]),  # 3.2x + 3.4y + 3.6z - 1.9
        None, ""
    ),
    (
        Expression(np.array([[1, 1.1], [1.1, 3]])),  # x**2 + 2.2x + 3
        np.array([2.2, 3]), None, ""
    ),
    (
        Expression(np.array([[1]])),  # 1
        np.array([1]), None, ""
    ),
    (
        Expression(np.array([[0, 0.5], [0.5, 0]])),  # x
        np.array([1, 0]), None, ""
    ),
    (
        Expression(None, '', 1, 1),  # y, in index, para way
        np.array([0, 1, 0]), None, ""
    )
    # (Expression(), np.array(), None, "")
])
def test_linear_complete_vector(input_data: Expression, expect: np.ndarray, expect_exception: _E,
                                expect_exception_match: str):
    if expect_exception is not None:
        with pytest.raises(expect_exception, match=expect_exception_match):
            assert np.array_equal(input_data.linear_complete_vector, expect)
    else:
        assert np.array_equal(input_data.linear_complete_vector, expect)


@pytest.mark.parametrize("a, b, expect, expect_exception, expect_exception_match", [
    (  # raise exception
        Expression(None, '<', 0, 1), Expression(None, '', 1, 1),
        None, NotImplementedError, "unsupported operate on constraint"
    ),
    (  # x+y
        Expression(None, '', 0, 1), Expression(None, '', 1, 1),  # x+y
        Expression(np.array([[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, 0]])), None, ""
    ),
    # (Expression(),Expression(),Expression(),None,""),
])
def test_add(a: Expression, b: Expression, expect: Expression, expect_exception: _E, expect_exception_match: str):
    if expect_exception is not None:
        with pytest.raises(expect_exception, match=expect_exception_match):
            assert (a + b).equal(expect)
    else:
        assert (a + b).equal(expect)


@pytest.mark.parametrize("a, b, expect, expect_exception, expect_exception_match", [])
def test_sub(a: Expression, b: Expression, expect: Expression, expect_exception: _E, expect_exception_match: str):
    pass


@pytest.mark.parametrize("a, b, expect, expect_exception, expect_exception_match", [])
def test_mul(a: Expression, b: Expression, expect: Expression, expect_exception: _E, expect_exception_match: str):
    pass


# @pytest.mark.parametrize("a, pow, expect, expect_exception, expect_exception_match", [])
# def test_pow():
#     pass
