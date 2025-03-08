import numpy as np
from typing import List, Tuple, Optional
from ..core.function import Function
from ..core.evaluator import safe_evaluate

def find_poles(
    func: Function, 
    search_range: Tuple[float, float] = (-10, 10), 
    precision: float = 1e-6, 
    num_samples: int = 1000
) -> List[float]:
    """
    Find the poles (singularities) of a function.
    
    This function looks for points where the function value approaches infinity.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Precision for finding poles
    :param num_samples: Number of points to sample in the search range
    :return: List of x-values where poles are located
    """
    start, end = search_range
    
    # Sample the function at regular intervals
    x_samples = np.linspace(start, end, num_samples)
    y_samples = np.array([safe_evaluate(func, x) for x in x_samples])
    
    # Find regions where the function becomes very large or undefined
    pole_candidates = []
    for i in range(len(y_samples)-1):
        if (np.isinf(y_samples[i]) or np.isnan(y_samples[i]) or 
            np.isinf(y_samples[i+1]) or np.isnan(y_samples[i+1])):
            # Narrow down the pole location by binary search
            pole = _refine_pole_location(func, x_samples[i], x_samples[i+1], precision)
            if pole is not None:
                pole_candidates.append(pole)
    
    # Remove duplicates (within precision)
    unique_poles = []
    for pole in pole_candidates:
        if not any(np.isclose(pole, p, atol=precision) for p in unique_poles):
            unique_poles.append(pole)
    
    return sorted(unique_poles)

def _refine_pole_location(
    func: Function, 
    a: float, 
    b: float, 
    precision: float, 
    max_iter: int = 20
) -> Optional[float]:
    """
    Refine the location of a pole using binary search.
    
    :param func: Function to analyze
    :param a: Left endpoint of the interval
    :param b: Right endpoint of the interval  
    :param precision: Desired precision
    :param max_iter: Maximum number of iterations
    :return: Approximation of the pole location, or None if no pole found
    """
    for _ in range(max_iter):
        # If interval is small enough, return midpoint
        if abs(b - a) < precision:
            return (a + b) / 2
            
        # Test the midpoint
        mid = (a + b) / 2
        y_mid = safe_evaluate(func, mid)
        
        # Test points slightly to the left and right
        delta = (b - a) * 0.001
        y_left = safe_evaluate(func, mid - delta)
        y_right = safe_evaluate(func, mid + delta)
        
        # Check if midpoint might be a pole
        if (np.isinf(y_mid) or np.isnan(y_mid)):
            return mid
            
        # Update the search interval based on where the function grows most rapidly
        left_change = abs(y_mid - y_left) if np.isfinite(y_mid) and np.isfinite(y_left) else np.inf
        right_change = abs(y_right - y_mid) if np.isfinite(y_right) and np.isfinite(y_mid) else np.inf
        
        if left_change > right_change:
            b = mid
        else:
            a = mid
    
    # If we've exhausted iterations without finding a clear pole
    return (a + b) / 2
