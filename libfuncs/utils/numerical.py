"""Numerical utility functions for the libfuncs library."""

import numpy as np
from typing import Callable, Union, List, Tuple, Optional

def linspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    """
    Return evenly spaced numbers over a specified interval.
    
    This is a wrapper around numpy.linspace.
    
    :param start: Start of interval
    :param stop: End of interval
    :param num: Number of samples to generate
    :return: Array of evenly spaced samples
    """
    return np.linspace(start, stop, num)

def find_nearest_value(array: np.ndarray, value: float) -> Tuple[float, int]:
    """
    Find the closest value in an array to a given value.
    
    :param array: Array to search in
    :param value: Value to find closest to
    :return: Tuple of (closest_value, index)
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the moving average of a dataset.
    
    :param data: Input data array
    :param window_size: Size of the moving window
    :return: Array with moving averages
    """
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def newton_method(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, int, bool]:
    """
    Find the root of a function using Newton's method.
    
    :param f: Function to find the root of
    :param df: Derivative of the function
    :param x0: Initial guess
    :param tol: Tolerance
    :param max_iter: Maximum number of iterations
    :return: Tuple of (root_approximation, iterations, converged)
    """
    x = x0
    for i in range(max_iter):
        f_val = f(x)
        if abs(f_val) < tol:
            return x, i, True
        
        df_val = df(x)
        if df_val == 0:
            return x, i, False
        
        x = x - f_val / df_val
    
    return x, max_iter, False

def interpolate_linear(
    x: np.ndarray, 
    y: np.ndarray, 
    x_new: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Perform linear interpolation.
    
    :param x: x-coordinates of data points
    :param y: y-coordinates of data points
    :param x_new: x-coordinates to interpolate at
    :return: Interpolated values
    """
    return np.interp(x_new, x, y)
