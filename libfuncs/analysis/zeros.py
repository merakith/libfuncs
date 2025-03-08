import numpy as np
from typing import List, Tuple, Optional
from ..core.function import Function
from ..core.evaluator import safe_evaluate

def find_zeros(
    func: Function, 
    search_range: Tuple[float, float] = (-10, 10),
    precision: float = 1e-6, 
    max_iterations: int = 100
) -> List[float]:
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
    start, end = search_range
    
    # For trigonometric functions, we need more samples to catch all zeros
    # Check if we're dealing with a trigonometric function
    is_trig = any(term in func.expr for term in ['sin', 'cos', 'tan'])
    
    # Sample the function at regular intervals to find potential zero regions
    num_samples = min(2000 if is_trig else 1000, int((end - start) / precision * 10))
    x_samples = np.linspace(start, end, num_samples)
    y_samples = np.array([safe_evaluate(func, x) for x in x_samples])
    
    # Find where the function changes sign (potential zero regions)
    sign_changes = []
    for i in range(len(y_samples)-1):
        if y_samples[i] * y_samples[i+1] <= 0 and np.isfinite(y_samples[i]) and np.isfinite(y_samples[i+1]):
            sign_changes.append((x_samples[i], x_samples[i+1]))
    
    # Use bisection method to refine each potential zero
    zeros = []
    for a, b in sign_changes:
        try:
            zero = bisection_method(func, a, b, precision, max_iterations)
            if zero is not None:
                zeros.append(zero)
        except Exception:
            # Skip if bisection fails
            pass
    
    # Remove duplicates (within precision)
    unique_zeros = []
    for zero in zeros:
        if not any(np.isclose(zero, z, atol=precision) for z in unique_zeros):
            unique_zeros.append(zero)
    
    return sorted(unique_zeros)

def bisection_method(
    func: Function, 
    a: float, 
    b: float, 
    precision: float = 1e-6, 
    max_iterations: int = 100
) -> Optional[float]:
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

def newton_method(
    func: Function,
    x0: float,
    precision: float = 1e-6,
    max_iterations: int = 100
) -> Optional[float]:
    """
    Find a root of the function using Newton's method.
    
    Newton's method uses the derivative to converge more quickly to a root.
    
    :param func: Function to find the root of
    :param x0: Initial guess
    :param precision: Desired precision
    :param max_iterations: Maximum number of iterations
    :return: Approximation of the root, or None if method fails
    """
    from ..calculus.derivatives import compute_derivative
    
    # Compute the derivative
    deriv = compute_derivative(func)
    
    x = x0
    for i in range(max_iterations):
        fx = safe_evaluate(func, x)
        
        # Check if we've found a root
        if abs(fx) < precision:
            return x
        
        # Compute derivative at current point
        dfx = safe_evaluate(deriv, x)
        
        # Check if derivative is too small (avoiding division by near-zero)
        if abs(dfx) < 1e-10:
            return None
        
        # Newton update: x_{n+1} = x_n - f(x_n)/f'(x_n)
        x_new = x - fx / dfx
        
        # Check for convergence
        if abs(x_new - x) < precision:
            return x_new
        
        x = x_new
    
    # If we reach here, we've hit max iterations without converging
    return None

def find_zeros_newton(
    func: Function, 
    search_range: Tuple[float, float] = (-10, 10),
    num_guesses: int = 20,
    precision: float = 1e-6,
    max_iterations: int = 100
) -> List[float]:
    """
    Find the zeros of a function using Newton's method with multiple starting points.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param num_guesses: Number of initial guesses to try
    :param precision: Desired precision for the roots
    :param max_iterations: Maximum number of iterations for Newton's method
    :return: List of x-values where the function is approximately zero
    """
    start, end = search_range
    
    # Generate initial guesses evenly spaced across the range
    x_guesses = np.linspace(start, end, num_guesses)
    
    # Apply Newton's method from each starting point
    zeros = []
    for x0 in x_guesses:
        zero = newton_method(func, x0, precision, max_iterations)
        if zero is not None and start <= zero <= end:
            zeros.append(zero)
    
    # Remove duplicates (within precision)
    unique_zeros = []
    for zero in zeros:
        if not any(np.isclose(zero, z, atol=precision) for z in unique_zeros):
            unique_zeros.append(zero)
    
    return sorted(unique_zeros)
