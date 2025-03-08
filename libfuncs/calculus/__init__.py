"""Calculus-related functionality for the libfuncs library."""

from .derivatives import (
    compute_derivative,
    numerical_derivative,
    symbolic_derivative,
    partial_derivative,
    gradient,
    directional_derivative,
)

from .integrals import (
    compute_integral,
    improper_integral, 
    numerical_antiderivative,
    trapezoid_rule,
    simpsons_rule,
    adaptive_simpsons_rule,
    monte_carlo_integration,
    gauss_quadrature,
    romberg_integration,
)

from .extrema import (
    find_extrema,
    find_inflection_points,
    find_zeros,
    find_local_extrema,
    find_global_extrema,
    classify_critical_point,
    concavity_analysis,
)

__all__ = [
    # Derivatives
    'compute_derivative',
    'numerical_derivative',
    'symbolic_derivative',
    'partial_derivative',
    'gradient',
    'directional_derivative',
    
    # Integrals
    'compute_integral',
    'improper_integral',
    'numerical_antiderivative',
    'trapezoid_rule',
    'simpsons_rule',
    'adaptive_simpsons_rule',
    'monte_carlo_integration',
    'gauss_quadrature',
    'romberg_integration',
    
    # Extrema
    'find_extrema',
    'find_inflection_points',
    'find_zeros',
    'find_local_extrema',
    'find_global_extrema',
    'classify_critical_point',
    'concavity_analysis',
]
