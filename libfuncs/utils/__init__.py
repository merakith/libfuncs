"""Utility functions for the libfuncs library."""

from .numerical import (
    linspace,
    find_nearest_value,
    moving_average,
    newton_method,
    interpolate_linear,
)

__all__ = [
    'linspace',
    'find_nearest_value',
    'moving_average',
    'newton_method',
    'interpolate_linear',
]
