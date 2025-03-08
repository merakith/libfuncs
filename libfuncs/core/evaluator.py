import numpy as np
from typing import Dict, Any, Union, List, Callable
import inspect

def evaluate_expression(compiled_expr: object, variables: Dict[str, Any]) -> Union[float, np.ndarray]:
    """
    Evaluate a compiled expression with provided variables.
    
    This function safely evaluates a compiled Python expression in a controlled
    namespace that includes only the necessary functions and variables.
    
    :param compiled_expr: A compiled Python expression (from compile())
    :param variables: Dictionary of variable names and their values
    :return: Result of evaluating the expression
    :raises ValueError: If evaluation fails
    """
    try:
        # Create a safe namespace with numpy functions and variables
        namespace = _create_safe_namespace()
        namespace.update(variables)
        
        # Evaluate the expression
        result = eval(compiled_expr, namespace)
        return result
    except ZeroDivisionError:
        return np.inf
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def _create_safe_namespace() -> Dict[str, Any]:
    """
    Create a safe namespace for evaluating mathematical expressions.
    
    This includes numpy functions and constants but excludes potentially
    dangerous built-ins.
    
    :return: Dictionary containing safe functions and constants
    """
    # Start with an empty namespace
    namespace = {}
    
    # Add numpy module
    namespace['np'] = np
    
    # Add selected numpy functions directly to namespace for convenience
    # Only include mathematical functions that are commonly used
    safe_np_funcs = [
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
        'exp', 'log', 'log10', 'log2', 'sqrt', 'cbrt',
        'abs', 'fabs', 'floor', 'ceil', 'round',
        'deg2rad', 'rad2deg', 'sign',
        'maximum', 'minimum', 'clip',
        'isfinite', 'isinf', 'isnan'
    ]
    
    for func_name in safe_np_funcs:
        if hasattr(np, func_name):
            namespace[func_name] = getattr(np, func_name)
    
    # Add numpy constants
    namespace['pi'] = np.pi
    namespace['e'] = np.e
    namespace['inf'] = np.inf
    namespace['nan'] = np.nan
    
    return namespace

def evaluate_over_range(func: Callable, start: float, end: float, points: int = 100) -> tuple:
    """
    Evaluate a function over a range of x values.
    
    :param func: Function to evaluate
    :param start: Start of range
    :param end: End of range
    :param points: Number of points to evaluate
    :return: Tuple of (x_values, y_values)
    """
    x_values = np.linspace(start, end, points)
    y_values = np.array([safe_evaluate(func, x) for x in x_values])
    return x_values, y_values

def safe_evaluate(func: Callable, *args, **kwargs) -> float:
    """
    Safely evaluate a function handling exceptions.
    
    :param func: Function to evaluate
    :param args: Positional arguments
    :param kwargs: Keyword arguments
    :return: Function value or np.nan if evaluation fails
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan

def estimate_domain(func: Callable, start: float = -100, end: float = 100, 
                   points: int = 1000) -> tuple:
    """
    Estimate the practical domain of a function by sampling.
    
    :param func: Function to evaluate
    :param start: Start of search range
    :param end: End of search range
    :param points: Number of points to evaluate
    :return: Tuple of (domain_start, domain_end)
    """
    x_vals = np.linspace(start, end, points)
    y_vals = np.array([safe_evaluate(func, x) for x in x_vals])
    
    # Find where function is defined (not nan, not inf)
    valid_indices = np.isfinite(y_vals)
    valid_x = x_vals[valid_indices]
    
    if len(valid_x) == 0:
        return (np.nan, np.nan)  # Function not defined in range
    
    return (np.min(valid_x), np.max(valid_x))

def check_continuity(func: Callable, point: float, 
                    epsilon: float = 1e-6) -> bool:
    """
    Check if a function is continuous at a given point.
    
    :param func: Function to check
    :param point: Point to check continuity at
    :param epsilon: Small value to use for limit calculation
    :return: True if function appears continuous, False otherwise
    """
    # Evaluate left and right limits
    left_vals = [func(point - epsilon * (0.1**i)) for i in range(1, 5)]
    right_vals = [func(point + epsilon * (0.1**i)) for i in range(1, 5)]
    
    # Remove any nan or inf values
    left_vals = [x for x in left_vals if np.isfinite(x)]
    right_vals = [x for x in right_vals if np.isfinite(x)]
    
    # If we don't have enough valid values, function might have a singularity
    if len(left_vals) < 2 or len(right_vals) < 2:
        return False
    
    # Check if left and right limits converge to the same value
    # Using tolerance relative to the values being compared
    left_limit = left_vals[-1]
    right_limit = right_vals[-1]
    
    tol = max(1e-10, epsilon * max(abs(left_limit), abs(right_limit)))
    return abs(left_limit - right_limit) < tol