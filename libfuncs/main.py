import numpy as np
import re


class Function:
    """
    A class for representing mathematical functions and performing operations on them.
    """

    def __init__(self, expr: str):
        """
        Constructor for Function class.

        :param expr: A string representation of the function.
        """
        self.expr = expr
        self.compiled_expr = self._compile_expr(expr)

    def __call__(self, **kwargs: float):
        """
        Evaluate the function at a given value.

        :param x: The value at which to evaluate the function.
        :type x: float
        :return: The value of the function at the given value.
        :rtype: float
        """
        try:
            print(kwargs)
            return eval(self.compiled_expr, kwargs, globals())
        except ZeroDivisionError:
            return "not defined"

    def _compile_expr(self, expr: str):
        print(expr, end="")
        expr = self._convert_expr(expr)
        print(" converted to " + expr)
        return compile(expr, "<string>", "eval")

    def _convert_expr(self, expr: str):
        # Replace pi and e with its value
        expr = re.sub(r"pi", str(np.pi), expr)
        expr = re.sub(r"e", str(np.e), expr)

        # Replace trigonometric functions
        expr = re.sub(r"sin\(([^)]+)\)", r"np.sin(np.deg2rad(\1))", expr)
        expr = re.sub(r"cos\(([^)]+)\)", r"np.cos(np.deg2rad(\1))", expr)
        expr = re.sub(r"tan\(([^)]+)\)", r"np.tan(np.deg2rad(\1))", expr)
        expr = re.sub(r"sec\(([^)]+)\)", r"1/np.cos(np.deg2rad(\1))", expr)
        expr = re.sub(r"cosec\(([^)]+)\)", r"1/np.sin(np.deg2rad(\1))", expr)
        expr = re.sub(r"cot\(([^)]+)\)", r"1/np.tan(np.deg2rad(\1))", expr)

        # Replace logarithmic functions
        expr = re.sub(r"log\(([^)]+)\)", r"np.log(\1)", expr)
        expr = re.sub(r"ln\(([^)]+)\)", r"np.log(\1)", expr)

        # Replace the caret (^) with ** for exponentiation
        expr = re.sub(r"\^", "**", expr)

        # Replace the mod symbol by absolute value function
        expr = re.sub(r"\|([^)]+)\|", r"np.fabs(\1)", expr)

        return expr

    def derivative(self, n=1):
        """
        Compute the nth derivative of the function.

        :param n: The order of the derivative to compute. Defaults to 1.
        :type n: int
        :return: A new Function object representing the nth derivative of the function.
        :rtype: Function
        """
        pass

    def integral(self, a, b):
        """
        Compute the definite integral of the function over the interval [a, b].

        :param a: The lower limit of integration.
        :type a: float
        :param b: The upper limit of integration.
        :type b: float
        :return: The value of the definite integral of the function over the interval [a, b].
        :rtype: float
        """
        pass

    def zeros(self):
        """
        Find the zeros of the function.

        :return: A list of values that are zeros of the function.
        :rtype: list of float
        """
        pass

    def poles(self):
        """
        Find the poles of the function.

        :return: A list of values that are poles of the function.
        :rtype: list of float
        """
        pass

    def roots(self):
        """
        Find the roots of the function (i.e. the values of x for which f(x) = 0).

        :return: A list of values that are roots of the function.
        :rtype: list of float
        """
        pass

    def extrema(self):
        """
        Find the extrema of the function (i.e. the values of x for which f'(x) = 0).

        :return: A dictionary containing the values of x at which the function has extrema and their types (maxima or minima).
        :rtype: dict
        """
        pass

    def inflection_points(self):
        """
        Find the inflection points of the function (i.e. the values of x at which the concavity of the function changes).

        :return: A list of values that are inflection points of the function.
        :rtype: list of float
        """
        pass

    def plot(self, x_range=(-10, 10), y_range=(-10, 10), resolution=1000):
        """
        Plot the function on the Cartesian plane.

        :param x_range: A tuple of two floats specifying the range of x values to plot. Defaults to (-10, 10).
        :type x_range: tuple of float
        :param y_range: A tuple of two floats specifying the range of y values to plot. Defaults to (-10, 10).
        :type y_range: tuple of float
        :param resolution: The number of points to use when plotting the function. Defaults to 1000.
        :type resolution: int
        """
        pass


function = Function
