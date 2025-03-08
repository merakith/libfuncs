from typing import List, Tuple
from ..core.function import Function
from .zeros import find_zeros

def find_roots(
    func: Function, 
    search_range: Tuple[float, float] = (-10, 10), 
    precision: float = 1e-6, 
    max_iterations: int = 100
) -> List[float]:
    """
    Find the roots of a function (values where f(x) = 0).
    
    This is an alias for find_zeros.
    
    :param func: Function to analyze
    :param search_range: Tuple of (start, end) for the search range
    :param precision: Desired precision for the roots
    :param max_iterations: Maximum number of iterations for the root-finding algorithm
    :return: List of x-values where the function equals zero
    """
    return find_zeros(func, search_range, precision, max_iterations)
