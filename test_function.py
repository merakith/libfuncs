import numpy as np
import pytest
from libfuncs import Function


def test_linear_function_evaluation():
    # Test evaluation of a linear function
    f = Function("2*x + 1")
    assert f(0) == 1
    assert f(1) == 3
    assert f(-1) == -1


def test_quadratic_function_evaluation():
    # Test evaluation of a quadratic function
    f = Function("x^2 + 2*x + 1")
    assert f(0) == 1
    assert f(1) == 4
    assert f(-1) == 0


def test_multivariable_function_evaluation():
    # Test evaluation of a function with many variables
    f = Function("2*x + y + 1")
    assert f(x=1, y=2) == 5
    assert f(x=0, y=0) == 1


def test_trigonometric_function_evaluation():
    # Test evaluation of a trigonometric function
    f = Function("sin(x)")
    assert f(0) == pytest.approx(0)
    assert f(90) == pytest.approx(1)
    assert f(180) == pytest.approx(0)
    assert f(270) == pytest.approx(-1)


def test_logarithmic_function_evaluation():
    # Test evaluation of a logarithmic function
    f = Function("log(x)")
    assert f(1) == 0
    assert f(np.e) == pytest.approx(1)


def test_mixed_function_evaluation():
    # Test evaluation of a function with a mixture of functions
    f = Function("(cos(x) + abs(x))/(x - 2)")
    assert f(0) == pytest.approx(1 / (-2))
    assert f(2) == np.inf
