import numpy as np
import inspect
from typing import Union, Dict, List, Tuple, Any, Optional, Callable

# These will be implemented in separate modules
from .parser import parse_expression, convert_expression
from .evaluator import evaluate_expression

class Function:
    """
    A class for representing mathematical functions and performing operations on them.
    
    This is the core class of the libfuncs library, providing a unified interface
    for working with mathematical functions, including evaluation, calculus operations,
    analysis, and visualization.
    """

    def __init__(self, expr: str):
        """
        Constructor for Function class.

        :param expr: A string representation of the function.
        :raises ValueError: If the expression cannot be parsed.
        """
        self.expr = expr
        self._original_expr = expr  # Store the original expression
        try:
            # These will call the functions from parser.py once implemented
            # For now we'll use placeholder methods
            self.compiled_expr = self._compile_expr(expr)
        except Exception as e:
            raise ValueError(f"Invalid expression: {expr}. Error: {e}")

    def __call__(self, *args, **kwargs) -> float:
        """
        Evaluate the function at given values.
        
        Can be called either with positional arguments (assuming x as variable)
        or with keyword arguments for multi-variable functions.
        
        Examples:
            f(2)       # Evaluates f at x=2
            f(x=2, y=3) # Evaluates f for x=2 and y=3
            
        :param args: Positional arguments (first arg is used as x)
        :param kwargs: Named arguments (variable names and values)
        :return: The value of the function at the given point(s)
        :rtype: float or numpy array
        :raises ValueError: For invalid input values
        """
        # Handle positional arguments (assume x as the variable)
        if args and not kwargs:
            kwargs = {'x': args[0]}
            
        try:
            # This will call evaluate_expression from evaluator.py once implemented
            # For now we use the direct eval method
            namespace = {'np': np}
            namespace.update(kwargs)
            return eval(self.compiled_expr, namespace)
        except ZeroDivisionError:
            return np.inf
        except Exception as e:
            raise ValueError(f"Error evaluating function at {kwargs}: {e}")

    def _compile_expr(self, expr: str):
        """
        Compile the expression string into a code object for efficient evaluation.
        
        :param expr: Mathematical expression as a string
        :return: Compiled code object
        """
        expr = self._convert_expr(expr)
        return compile(expr, "<string>", "eval")

    def _convert_expr(self, expr: str) -> str:
        """
        Convert mathematical notation to Python code.
        
        :param expr: Mathematical expression as a string
        :return: Python expression as a string
        """
        # Replace pi and e with their values
        expr = expr.replace("pi", str(np.pi))
        expr = expr.replace("e", str(np.e))

        # Replace trigonometric functions
        import re
        expr = re.sub(r"sin\(([^)]+)\)", r"np.sin(np.radians(\1))", expr)
        expr = re.sub(r"cos\(([^)]+)\)", r"np.cos(np.radians(\1))", expr)
        expr = re.sub(r"tan\(([^)]+)\)", r"np.tan(np.radians(\1))", expr)
        expr = re.sub(r"sec\(([^)]+)\)", r"1/np.cos(np.radians(\1))", expr)
        expr = re.sub(r"cosec\(([^)]+)\)", r"1/np.sin(np.radians(\1))", expr)
        expr = re.sub(r"cot\(([^)]+)\)", r"1/np.tan(np.radians(\1))", expr)

        # Replace logarithmic functions
        expr = re.sub(r"log\(([^)]+)\)", r"np.log10(\1)", expr)
        expr = re.sub(r"ln\(([^)]+)\)", r"np.log(\1)", expr)

        # Replace the caret (^) with ** for exponentiation
        expr = re.sub(r"\^", "**", expr)

        # Replace the mod symbol by absolute value function
        expr = re.sub(r"\|([^|]+)\|", r"np.fabs(\1)", expr)

        return expr

    def __str__(self) -> str:
        """
        Return a string representation of the function.
        
        :return: String representation
        """
        if hasattr(self, '_original_expr') and self._original_expr != self.expr:
            return f"Function('{self._original_expr}')"
        return f"Function('{self.expr}')"
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the function.
        
        :return: String representation
        """
        if hasattr(self, '_original_expr') and self._original_expr != self.expr:
            return f"Function('{self._original_expr}')"
        return f"Function('{self.expr}')"
        
    # Methods that delegate to specialized modules
    
    def derivative(self, n: int = 1):
        """
        Compute the nth derivative of the function.

        :param n: The order of the derivative to compute. Defaults to 1.
        :type n: int
        :return: A new Function object representing the nth derivative of the function.
        :rtype: Function
        """
        from ..calculus.derivatives import compute_derivative
        return compute_derivative(self, n)

    def integral(self, a: float, b: float) -> float:
        """
        Compute the definite integral of the function over the interval [a, b].

        :param a: The lower limit of integration.
        :type a: float
        :param b: The upper limit of integration.
        :type b: float
        :return: The value of the definite integral of the function over the interval [a, b].
        :rtype: float
        """
        from ..calculus.integrals import compute_integral
        return compute_integral(self, a, b)

    def zeros(self) -> List[float]:
        """
        Find the zeros of the function.

        :return: A list of values that are zeros of the function.
        :rtype: list of float
        """
        from ..analysis.zeros import find_zeros
        return find_zeros(self)

    def poles(self) -> List[float]:
        """
        Find the poles of the function.

        :return: A list of values that are poles of the function.
        :rtype: list of float
        """
        from ..analysis.poles import find_poles
        return find_poles(self)

    def roots(self) -> List[float]:
        """
        Find the roots of the function (i.e. the values of x for which f(x) = 0).

        :return: A list of values that are roots of the function.
        :rtype: list of float
        """
        from ..analysis.roots import find_roots
        return find_roots(self)

    def extrema(self) -> Dict[float, str]:
        """
        Find the extrema of the function (i.e. the values of x for which f'(x) = 0).

        :return: A dictionary containing the values of x at which the function has extrema and their types (maxima or minima).
        :rtype: dict
        """
        from ..calculus.extrema import find_extrema
        return find_extrema(self)

    def inflection_points(self) -> List[float]:
        """
        Find the inflection points of the function (i.e. the values of x at which the concavity of the function changes).

        :return: A list of values that are inflection points of the function.
        :rtype: list of float
        """
        from ..calculus.extrema import find_inflection_points
        return find_inflection_points(self)

    def plot(self, x_range: Tuple[float, float] = (-10, 10), 
             y_range: Optional[Tuple[float, float]] = None, 
             resolution: int = 1000,
             show_plot: bool = True):
        """
        Plot the function on the Cartesian plane.

        :param x_range: A tuple of two floats specifying the range of x values to plot. Defaults to (-10, 10).
        :type x_range: tuple of float
        :param y_range: A tuple of two floats specifying the range of y values to plot. Defaults to (-10, 10).
        :type y_range: tuple of float
        :param resolution: The number of points to use when plotting the function. Defaults to 1000.
        :type resolution: int
        :param show_plot: Whether to display the plot
        :type show_plot: bool
        """
        from ..visualization.plotting import plot_function
        return plot_function(self, x_range, y_range, resolution=resolution, show_plot=show_plot)