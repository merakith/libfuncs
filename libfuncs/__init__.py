"""This library aims to provide easier calculation to problems related to mathematical functions."""


__title__ = "libfuncs"
__author__ = "ItsMrNatural"
__license__ = "MIT"
__version__ = "0.1.0"


# Import main functionality 
from .core.function import Function

# Import key calculus operations
from .calculus.derivatives import compute_derivative
from .calculus.integrals import compute_integral
from .calculus.extrema import find_extrema, find_inflection_points

# Import analysis functionality
from .analysis import find_zeros, find_poles, find_roots

# Import visualization
from .visualization.plotting import (
    plot_function, 
    plot_derivative,
    plot_with_extrema,
    plot_integral,
    plot_multiple_functions,
    plot_3d_function,
    plot_contour,
    plot_vector_field
)

# Define what gets imported with "from libfuncs import *"
__all__ = [
    'Function',
    'compute_derivative',
    'compute_integral',
    'find_extrema',
    'find_inflection_points',
    'find_zeros',
    'find_poles',
    'find_roots',
    'plot_function',
    'plot_derivative',
    'plot_with_extrema',
    'plot_integral',
    'plot_multiple_functions',
    'plot_3d_function',
    'plot_contour',
    'plot_vector_field',
]
