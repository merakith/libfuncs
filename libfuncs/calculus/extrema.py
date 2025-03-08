import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from ..core.function import Function
from ..core.evaluator import safe_evaluate, evaluate_over_range
from .derivatives import compute_derivative, numerical_derivative

def find_extrema(func: Function, search_range: Tuple[float, float] = (-10, 10), 
               precision: float = 1e-4) -> Dict[float, str]:
    """
    Find the extrema (maxima and minima) of a function.
    
    Uses the first and second derivatives to identify and classify critical points.
    
    :param func: Function object to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Tolerance for finding zeros of the derivative
    :return: Dictionary mapping x-values to extrema type ('maxima', 'minima', or 'inflection')
    """
    # Compute first and second derivatives
    first_deriv = compute_derivative(func, n=1)
    second_deriv = compute_derivative(func, n=2)
    
    # Find critical points (where first derivative is zero)
    from ..analysis.zeros import find_zeros
    critical_points = find_zeros(first_deriv, search_range, precision)
    
    # Analyze each critical point
    extrema = {}
    for x in critical_points:
        # Evaluate second derivative at critical point
        try:
            second_deriv_value = second_deriv(x)
            
            # Classify the critical point based on the second derivative
            if np.isclose(second_deriv_value, 0, atol=precision):
                extrema[x] = 'inflection/saddle'
            elif second_deriv_value > 0:
                extrema[x] = 'minima'
            else:  # second_deriv_value < 0
                extrema[x] = 'maxima'
        except Exception:
            # Skip points where second derivative can't be evaluated
            continue
    
    return extrema

def find_inflection_points(func: Function, search_range: Tuple[float, float] = (-10, 10), 
                         precision: float = 1e-4) -> List[float]:
    """
    Find the inflection points of a function.
    
    Inflection points occur where the second derivative changes sign.
    
    :param func: Function object to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Tolerance for finding zeros of the second derivative
    :return: List of x-values where inflection occurs
    """
    # Compute second derivative
    second_deriv = compute_derivative(func, n=2)
    
    # Find zeros of second derivative
    inflection_candidates = find_zeros(second_deriv, search_range, precision)
    
    # Verify that the second derivative changes sign
    third_deriv = compute_derivative(func, n=3)
    
    # Filter out points where the third derivative is zero
    inflection_points = [x for x in inflection_candidates 
                       if abs(third_deriv(x)) > precision]
    
    return inflection_points

def find_zeros(func: Function, search_range: Tuple[float, float] = (-10, 10),
             precision: float = 1e-6, max_iterations: int = 100) -> List[float]:
    """
    Find the zeros of a function using a combination of methods.
    
    This function uses a sampling approach to find candidate regions,
    then refines the zeros using the bisection method.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Desired precision for the roots
    :param max_iterations: Maximum number of iterations for the root-finding algorithm
    :return: List of x-values where the function is approximately zero
    """
    # Import here to avoid circular imports
    from ..analysis.zeros import find_zeros as analyze_zeros
    return analyze_zeros(func, search_range, precision, max_iterations)

def bisection_method(func: Function, a: float, b: float, 
                   precision: float = 1e-6, max_iterations: int = 100) -> Optional[float]:
    """
    Find a root of the function using the bisection method.
    
    :param func: Function to find the root of
    :param a: Left endpoint of the interval
    :param b: Right endpoint of the interval
    :param precision: Desired precision
    :param max_iterations: Maximum number of iterations
    :return: Approximation of the root, or None if method fails
    """
    fa = safe_evaluate(func, a)
    fb = safe_evaluate(func, b)
    
    # Check if a or b is already a root
    if np.isclose(fa, 0, atol=precision):
        return a
    if np.isclose(fb, 0, atol=precision):
        return b
    
    # Ensure the function changes sign in the interval
    if fa * fb > 0:
        return None
    
    # Bisection algorithm
    for _ in range(max_iterations):
        # Compute midpoint
        c = (a + b) / 2
        fc = safe_evaluate(func, c)
        
        # Check if we found the root
        if np.isclose(fc, 0, atol=precision) or (b - a) / 2 < precision:
            return c
        
        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    # Return the midpoint if we reached max iterations
    return (a + b) / 2

def classify_critical_point(func: Function, x: float, 
                          precision: float = 1e-6) -> str:
    """
    Classify a critical point as maxima, minima, or inflection point.
    
    :param func: Function to analyze
    :param x: Critical point to classify
    :param precision: Precision for comparisons
    :return: Classification ('maxima', 'minima', 'inflection', or 'unknown')
    """
    # Compute second derivative at the point
    second_deriv = compute_derivative(func, n=2)
    second_deriv_value = second_deriv(x)
    
    # Classify based on the second derivative
    if np.isclose(second_deriv_value, 0, atol=precision):
        # Need to check higher derivatives
        third_deriv = compute_derivative(func, n=3)
        third_deriv_value = third_deriv(x)
        
        if np.isclose(third_deriv_value, 0, atol=precision):
            fourth_deriv = compute_derivative(func, n=4)
            fourth_deriv_value = fourth_deriv(x)
            
            if np.isclose(fourth_deriv_value, 0, atol=precision):
                return 'unknown'  # Higher order analysis needed
            elif fourth_deriv_value > 0:
                return 'minima'
            else:
                return 'maxima'
        else:
            return 'inflection'
    elif second_deriv_value > 0:
        return 'minima'
    else:  # second_deriv_value < 0
        return 'maxima'

def find_local_extrema(func: Function, search_range: Tuple[float, float] = (-10, 10),
                     precision: float = 1e-4) -> Dict[str, List[float]]:
    """
    Find the local extrema of a function.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Precision for finding extrema
    :return: Dictionary with 'maxima' and 'minima' lists
    """
    # Find all critical points and classify them
    extrema_dict = find_extrema(func, search_range, precision)
    
    # Separate into maxima and minima
    result = {'maxima': [], 'minima': [], 'inflection': []}
    
    for x, point_type in extrema_dict.items():
        if point_type in result:
            result[point_type].append(x)
    
    # Sort the lists
    for key in result:
        result[key].sort()
    
    return result

def find_global_extrema(func: Function, search_range: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
    """
    Find the global extrema of a function in a given range.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :return: Dictionary with 'max' and 'min' tuples (x, f(x))
    """
    # Find local extrema
    local_extrema = find_local_extrema(func, search_range)
    
    # Combine all candidate points (local extrema and endpoints)
    candidates = (
        local_extrema['maxima'] + 
        local_extrema['minima'] + 
        [search_range[0], search_range[1]]
    )
    
    # Evaluate the function at all candidate points
    values = [safe_evaluate(func, x) for x in candidates]
    
    # Filter out any non-finite values
    valid_points = [(x, y) for x, y in zip(candidates, values) if np.isfinite(y)]
    
    if not valid_points:
        return {
            'max': (None, None),
            'min': (None, None)
        }
    
    # Find the maximum and minimum
    valid_points.sort(key=lambda p: p[1])
    min_point = valid_points[0]
    max_point = valid_points[-1]
    
    return {
        'max': max_point,
        'min': min_point
    }

def concavity_analysis(func: Function, search_range: Tuple[float, float] = (-10, 10),
                     num_points: int = 100) -> Dict[str, List[Tuple[float, float]]]:
    """
    Analyze the concavity of a function.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the analysis range
    :param num_points: Number of sample points
    :return: Dictionary with 'concave_up' and 'concave_down' regions
    """
    # Compute second derivative
    second_deriv = compute_derivative(func, n=2)
    
    # Sample the second derivative
    x_vals = np.linspace(search_range[0], search_range[1], num_points)
    y_vals = [safe_evaluate(second_deriv, x) for x in x_vals]
    
    # Find inflection points
    inflection_points = find_inflection_points(func, search_range)
    
    # Determine concavity regions
    concave_up = []
    concave_down = []
    
    # Add the endpoints and inflection points to create boundaries
    boundaries = sorted([search_range[0], search_range[1]] + inflection_points)
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mid = (start + end) / 2
        
        # Test the concavity at the midpoint of each region
        concavity = safe_evaluate(second_deriv, mid)
        
        if concavity > 0:
            concave_up.append((start, end))
        elif concavity < 0:
            concave_down.append((start, end))
    
    return {
        'concave_up': concave_up,
        'concave_down': concave_down
    }