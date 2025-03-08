import pytest
import numpy as np
from libfuncs.core.function import Function
from libfuncs.core.evaluator import safe_evaluate, evaluate_expression, evaluate_over_range
from libfuncs.core.parser import parse_expression, extract_variables, validate_expression

def test_function_methods():
    """Test that all Function class methods work as expected"""
    f = Function("x^2 - 4")
    
    # Test derivative method
    df = f.derivative()
    assert df(2) == pytest.approx(4, abs=1e-5)
    
    # Test integral method
    integral = f.integral(-2, 2)
    assert integral == pytest.approx(-10.67, abs=0.1)
    
    # Test zeros method
    zeros = f.zeros()
    assert len(zeros) == 2
    assert -2.0 in [pytest.approx(z, abs=1e-5) for z in zeros]
    assert 2.0 in [pytest.approx(z, abs=1e-5) for z in zeros]
    
    # Test extrema method
    extrema = f.extrema()
    assert len(extrema) == 1
    assert 0.0 in [pytest.approx(x, abs=1e-5) for x in extrema.keys()]
    
    # Test that plot method doesn't raise errors
    try:
        f.plot(show_plot=False)
    except Exception as e:
        pytest.fail(f"plot() method raised an exception: {e}")

def test_safe_evaluate():
    """Test the safe_evaluate function"""
    def div_by_zero(x):
        return 1 / x
    
    # Should not raise exception
    result = safe_evaluate(div_by_zero, 0)
    assert np.isnan(result)
    
    # Normal case
    result = safe_evaluate(div_by_zero, 2)
    assert result == 0.5

def test_parse_expression():
    """Test expression parsing"""
    expr = "x^2 + sin(y)"
    parsed = parse_expression(expr)
    
    # Should be a code object
    assert parsed is not None
    
    # Should evaluate correctly
    namespace = {'x': 2, 'y': 90, 'np': np}
    result = eval(parsed, namespace)
    assert result == pytest.approx(5, abs=1e-5)

def test_extract_variables():
    """Test variable extraction from expressions"""
    expr = "3*x + y*z - sin(t)"
    vars = extract_variables(expr)
    
    assert sorted(vars) == ['t', 'x', 'y', 'z']

def test_validate_expression():
    """Test expression validation"""
    # Valid expression
    valid, msg = validate_expression("x^2 + 3*x + 1")
    assert valid
    
    # Invalid - unbalanced parentheses
    valid, msg = validate_expression("x^2 + sin(x")
    assert not valid
    assert "parentheses" in msg.lower()
