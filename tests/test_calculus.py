import pytest
import numpy as np
from libfuncs import Function
from libfuncs.calculus import compute_derivative, compute_integral, find_extrema

def test_derivative_polynomial():
    """Test derivatives of polynomial functions"""
    # f(x) = x^2 + 2x + 1 => f'(x) = 2x + 2
    f = Function("x^2 + 2*x + 1")
    df = compute_derivative(f)
    
    assert df(0) == pytest.approx(2.0, abs=1e-5)
    assert df(1) == pytest.approx(4.0, abs=1e-5)
    assert df(-1) == pytest.approx(0.0, abs=1e-5)

def test_derivative_trigonometric():
    """Test derivatives of trigonometric functions"""
    # f(x) = sin(x) => f'(x) = cos(x)
    f = Function("sin(x)")
    df = compute_derivative(f)
    
    assert df(0) == pytest.approx(1.0, abs=1e-5)  # cos(0) = 1
    assert df(90) == pytest.approx(0.0, abs=1e-5)  # cos(90) = 0
    assert df(180) == pytest.approx(-1.0, abs=1e-5)  # cos(180) = -1

def test_higher_order_derivative():
    """Test higher-order derivatives"""
    # f(x) = x^3 => f''(x) = 6x
    f = Function("x^3")
    d2f = compute_derivative(f, n=2)
    
    assert d2f(0) == pytest.approx(0.0, abs=1e-5)
    assert d2f(1) == pytest.approx(6.0, abs=1e-5)
    assert d2f(-1) == pytest.approx(-6.0, abs=1e-5)

def test_integral_polynomial():
    """Test integration of polynomial functions"""
    # ∫(2x + 1)dx from 0 to 1 = [x^2 + x]_0^1 = 2
    f = Function("2*x + 1")
    integral_value = compute_integral(f, 0, 1)
    assert integral_value == pytest.approx(2.0, abs=1e-5)
    
    # ∫x^2 dx from 0 to 2 = [x^3/3]_0^2 = 8/3
    f = Function("x^2")
    integral_value = compute_integral(f, 0, 2)
    assert integral_value == pytest.approx(8/3, abs=1e-5)

def test_find_extrema():
    """Test finding extrema of functions"""
    # f(x) = x^2 has a minimum at x = 0
    f = Function("x^2")
    extrema = find_extrema(f, (-5, 5))
    
    # Test that we have found at least one extremum
    assert len(extrema) > 0, "No extrema found"
    
    # Check that one of the extrema is near x=0 (minimum)
    found_minimum = False
    for x, ext_type in extrema.items():
        if abs(x) < 0.1 and "min" in ext_type.lower():
            found_minimum = True
            break
    assert found_minimum, "Minimum at x=0 not found"
    
    # f(x) = -x^2 + 4 has a maximum at x = 0
    f = Function("-x^2 + 4")
    extrema = find_extrema(f, (-5, 5))
    
    # Test that we have found at least one extremum
    assert len(extrema) > 0, "No extrema found"
    
    # Check that one of the extrema is near x=0 (maximum)
    found_maximum = False
    for x, ext_type in extrema.items():
        if abs(x) < 0.1 and "max" in ext_type.lower():
            found_maximum = True
            break
    assert found_maximum, "Maximum at x=0 not found"
    
    # f(x) = x^3 has an inflection point at x = 0
    f = Function("x^3")
    extrema = find_extrema(f, (-5, 5))
    assert len(extrema) == 1
    assert 0.0 in [pytest.approx(x, abs=1e-5) for x in extrema.keys()]
    assert "inflection" in extrema[list(extrema.keys())[0]]
