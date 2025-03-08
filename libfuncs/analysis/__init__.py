"""Analysis-related functionality for the libfuncs library."""

from .zeros import find_zeros, find_zeros_newton
from .poles import find_poles
from .roots import find_roots

__all__ = [
    'find_zeros',
    'find_zeros_newton',
    'find_poles',
    'find_roots',
]
