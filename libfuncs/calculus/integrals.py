import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Dict, Any
from ..core.function import Function
from ..core.evaluator import safe_evaluate


def compute_integral(
    func: Function,
    a: float,
    b: float,
    method: str = "adaptive_simpson",
    tol: float = 1e-6,
    max_recursion: int = 20,
) -> float:
    """
    Compute the definite integral of a function over the interval [a, b].

    :param func: Function object to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param method: Integration method ('trapezoid', 'simpson', 'adaptive_simpson', 'gauss_quadrature', 'monte_carlo')
    :param tol: Error tolerance for adaptive methods
    :param max_recursion: Maximum recursion depth for adaptive methods
    :return: Approximate value of the definite integral
    :raises ValueError: If method is invalid or limits are invalid
    """
    # Check for valid integration limits
    if a > b:
        # Swap limits and negate result
        return -compute_integral(func, b, a, method, tol, max_recursion)

    # If limits are equal, integral is zero
    if a == b:
        return 0.0

    # Handle discontinuities or singularities at endpoints
    # by slightly adjusting the integration limits
    epsilon = (b - a) * 1e-10
    a_adjusted = a + epsilon
    b_adjusted = b - epsilon

    # Create a wrapper function for numerical integration
    def f(x):
        return safe_evaluate(func, x)

    # Choose integration method
    if method == "trapezoid":
        return trapezoid_rule(f, a_adjusted, b_adjusted, n=1000)

    elif method == "simpson":
        return simpsons_rule(f, a_adjusted, b_adjusted, n=500)

    elif method == "adaptive_simpson":
        return adaptive_simpsons_rule(f, a_adjusted, b_adjusted, tol, max_recursion)

    elif method == "gauss_quadrature":
        return gauss_quadrature(f, a_adjusted, b_adjusted, n=10)

    elif method == "monte_carlo":
        return monte_carlo_integration(f, a_adjusted, b_adjusted, n=100000)

    else:
        raise ValueError(f"Unknown integration method: {method}")


def trapezoid_rule(
    f: Callable[[float], float], a: float, b: float, n: int = 1000
) -> float:
    """
    Compute the definite integral using the trapezoid rule.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param n: Number of subintervals
    :return: Approximate value of the definite integral
    """
    # Width of each subinterval
    h = (b - a) / n

    # Sum the trapezoid areas
    result = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        result += f(x)

    return h * result


def simpsons_rule(
    f: Callable[[float], float], a: float, b: float, n: int = 500
) -> float:
    """
    Compute the definite integral using Simpson's rule.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param n: Number of subintervals (must be even)
    :return: Approximate value of the definite integral
    """
    # Ensure n is even
    if n % 2 != 0:
        n += 1

    # Width of each subinterval
    h = (b - a) / n

    # Apply Simpson's rule
    result = f(a) + f(b)

    # Sum the odd-indexed points (coefficient 4)
    for i in range(1, n, 2):
        x = a + i * h
        result += 4 * f(x)

    # Sum the even-indexed points (coefficient 2)
    for i in range(2, n, 2):
        x = a + i * h
        result += 2 * f(x)

    return (h / 3) * result


def adaptive_simpsons_rule(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_recursion: int = 20,
) -> float:
    """
    Compute the definite integral using adaptive Simpson's rule.

    This method recursively subdivides the interval until the
    desired tolerance is achieved.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param tol: Error tolerance
    :param max_recursion: Maximum recursion depth
    :return: Approximate value of the definite integral
    """

    def _adaptive_simpson_helper(
        a: float, b: float, fa: float, fm: float, fb: float, tol: float, depth: int
    ) -> float:
        """Recursive helper function for adaptive Simpson's rule."""
        # Compute midpoints for the two halves
        m = (a + b) / 2
        lm = (a + m) / 2
        rm = (m + b) / 2

        # Compute function values at the new midpoints
        flm = f(lm)
        frm = f(rm)

        # Compute Simpson approximations for the whole interval and two halves
        whole_integral = (b - a) * (fa + 4 * fm + fb) / 6
        left_integral = (m - a) * (fa + 4 * flm + fm) / 6
        right_integral = (b - m) * (fm + 4 * frm + fb) / 6

        # Estimate the error
        error = abs(left_integral + right_integral - whole_integral)

        # If error is small enough or we've reached max depth, return sum of halves
        if (error <= 15 * tol) or (depth >= max_recursion):
            return left_integral + right_integral

        # Otherwise, recursively compute each half with tighter tolerance
        return _adaptive_simpson_helper(
            a, m, fa, flm, fm, tol / 2, depth + 1
        ) + _adaptive_simpson_helper(m, b, fm, frm, fb, tol / 2, depth + 1)

    # Compute function values at endpoints and midpoint
    fa = f(a)
    fb = f(b)
    m = (a + b) / 2
    fm = f(m)

    # Start the recursion
    return _adaptive_simpson_helper(a, b, fa, fm, fb, tol, 0)


def gauss_quadrature(
    f: Callable[[float], float], a: float, b: float, n: int = 10
) -> float:
    """
    Compute the definite integral using Gaussian quadrature.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param n: Number of quadrature points
    :return: Approximate value of the definite integral
    """
    # Legendre polynomial roots and weights for n=10
    # These are precomputed for efficiency
    if n == 10:
        # Nodes (x) and weights (w) for Gauss-Legendre quadrature
        x = np.array(
            [
                -0.9739065285171717,
                -0.8650633666889845,
                -0.6794095682990244,
                -0.4333953941292472,
                -0.1488743389816312,
                0.1488743389816312,
                0.4333953941292472,
                0.6794095682990244,
                0.8650633666889845,
                0.9739065285171717,
            ]
        )

        w = np.array(
            [
                0.0666713443086881,
                0.1494513491505806,
                0.2190863625159820,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.2190863625159820,
                0.1494513491505806,
                0.0666713443086881,
            ]
        )
    else:
        # For other n values, use numpy's built-in legendre polynomial
        from numpy.polynomial.legendre import leggauss

        x, w = leggauss(n)

    # Transform from [-1, 1] to [a, b]
    t = 0.5 * (x + 1) * (b - a) + a

    # Compute the integral
    return 0.5 * (b - a) * sum(w[i] * f(t[i]) for i in range(n))


def monte_carlo_integration(
    f: Callable[[float], float], a: float, b: float, n: int = 100000
) -> float:
    """
    Compute the definite integral using Monte Carlo integration.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param n: Number of random points to sample
    :return: Approximate value of the definite integral
    """
    # Generate n random points in the interval [a, b]
    x_random = a + (b - a) * np.random.random(n)

    # Evaluate the function at these points
    f_values = np.array([f(x) for x in x_random])

    # Compute the mean value and multiply by the interval width
    mean_value = np.mean(f_values)
    result = (b - a) * mean_value

    return result


def improper_integral(
    func: Function, a: float, b: float, method: str = "limit", tol: float = 1e-6
) -> float:
    """
    Compute an improper integral (with infinite limits or singularities).

    :param func: Function object to integrate
    :param a: Lower limit of integration (can be -np.inf)
    :param b: Upper limit of integration (can be np.inf)
    :param method: Method for handling improper integrals ('limit' or 'transform')
    :param tol: Error tolerance
    :return: Approximate value of the improper integral
    :raises ValueError: If the integral does not converge
    """
    # Handle infinite limits
    if np.isinf(a) or np.isinf(b):
        return _compute_infinite_integral(func, a, b, method, tol)

    # Test for singularities at the endpoints
    fa = safe_evaluate(func, a)
    fb = safe_evaluate(func, b)

    if np.isinf(fa) or np.isnan(fa) or np.isinf(fb) or np.isnan(fb):
        # Use limit approach for singularity at endpoint
        return _compute_singularity_integral(func, a, b, method, tol)

    # If no singularities detected, use standard integration
    return compute_integral(func, a, b)


def _compute_infinite_integral(
    func: Function, a: float, b: float, method: str, tol: float
) -> float:
    """
    Compute an integral with infinite limits.

    :param func: Function to integrate
    :param a: Lower limit
    :param b: Upper limit
    :param method: Method to use
    :param tol: Tolerance
    :return: Approximate value of the improper integral
    """
    if method == "limit":
        # Use limit approach with finite cutoffs
        if a == -np.inf and b == np.inf:
            # Split at 0 and compute each half
            return _compute_infinite_integral(
                func, -np.inf, 0, method, tol / 2
            ) + _compute_infinite_integral(func, 0, np.inf, method, tol / 2)

        elif a == -np.inf:
            # Try increasingly negative values until integral converges
            result = 0
            cutoff = b - 1
            prev_result = np.inf

            while abs(result - prev_result) > tol:
                prev_result = result
                cutoff *= 2  # Double the cutoff each time
                result = compute_integral(func, -cutoff, b)

                if cutoff < -1e10:  # Prevent excessive computation
                    raise ValueError("Integral does not appear to converge")

            return result

        else:  # b == np.inf
            # Try increasingly large values until integral converges
            result = 0
            cutoff = a + 1
            prev_result = np.inf

            while abs(result - prev_result) > tol:
                prev_result = result
                cutoff *= 2  # Double the cutoff each time
                result = compute_integral(func, a, cutoff)

                if cutoff > 1e10:  # Prevent excessive computation
                    raise ValueError("Integral does not appear to converge")

            return result

    elif method == "transform":
        # Use variable transformation
        if a == -np.inf and b == np.inf:
            # Use tan substitution: x = tan(t) transforms (-inf, inf) to (-π/2, π/2)
            def transformed_func(t):
                x = np.tan(t)
                return func(x) * (1 + x**2)  # Include the Jacobian factor

            return compute_integral(transformed_func, -np.pi / 2 + tol, np.pi / 2 - tol)

        elif a == -np.inf:
            # Use substitution x = a + 1/t to map (-inf, b) to (0, 1/(b-a))
            def transformed_func(t):
                if t == 0:
                    return 0  # Avoid division by zero
                x = b - 1 / t
                return func(x) / t**2  # Include the Jacobian factor

            return compute_integral(transformed_func, 0, 1 / (b - a))

        else:  # b == np.inf
            # Use substitution x = a + 1/t to map (a, inf) to (0, 1/(b-a))
            def transformed_func(t):
                if t == 0:
                    return 0  # Avoid division by zero
                x = a + 1 / t
                return func(x) / t**2  # Include the Jacobian factor

            return compute_integral(transformed_func, 0, 1)

    else:
        raise ValueError(f"Unknown method for improper integral: {method}")


def _compute_singularity_integral(
    func: Function, a: float, b: float, method: str, tol: float
) -> float:
    """
    Compute an integral with singularities at the endpoints.

    :param func: Function to integrate
    :param a: Lower limit
    :param b: Upper limit
    :param method: Method to use
    :param tol: Tolerance
    :return: Approximate value of the improper integral
    """
    if method == "limit":
        # Compute from both endpoints, approaching the singularity

        # Test for singularity at a
        fa = safe_evaluate(func, a)
        singularity_at_a = np.isinf(fa) or np.isnan(fa)

        # Test for singularity at b
        fb = safe_evaluate(func, b)
        singularity_at_b = np.isinf(fb) or np.isnan(fb)

        if singularity_at_a and singularity_at_b:
            # Singularities at both endpoints - split at midpoint
            mid = (a + b) / 2
            return _compute_singularity_integral(
                func, a, mid, method, tol / 2
            ) + _compute_singularity_integral(func, mid, b, method, tol / 2)

        elif singularity_at_a:
            # Singularity at lower endpoint
            epsilon = (b - a) / 10
            result = 0
            prev_result = np.inf

            while abs(result - prev_result) > tol and epsilon > 1e-10:
                prev_result = result
                epsilon /= 2  # Halve the distance each time
                result = compute_integral(func, a + epsilon, b)

            return result

        else:  # singularity_at_b
            # Singularity at upper endpoint
            epsilon = (b - a) / 10
            result = 0
            prev_result = np.inf

            while abs(result - prev_result) > tol and epsilon > 1e-10:
                prev_result = result
                epsilon /= 2  # Halve the distance each time
                result = compute_integral(func, a, b - epsilon)

            return result

    elif method == "transform":
        # Transform the integral to remove singularities
        # This would need more sophisticated analysis of the singularity type
        # We'll use the limit approach as a fallback
        return _compute_singularity_integral(func, a, b, "limit", tol)

    else:
        raise ValueError(f"Unknown method for improper integral: {method}")


def numerical_antiderivative(
    func: Function, x0: float = 0.0, c: float = 0.0
) -> Function:
    """
    Create a numerical approximation of the antiderivative of a function.

    :param func: Function to integrate
    :param x0: Reference point (F(x0) = c)
    :param c: Value of the antiderivative at x0
    :return: Function object representing the antiderivative
    """

    # Create a function that computes the antiderivative at a point
    def antiderivative_func(x):
        if x == x0:
            return c
        elif x < x0:
            return c - compute_integral(func, x, x0)
        else:
            return c + compute_integral(func, x0, x)

    # Create a new Function object
    antiderivative_expr = f"∫({func.expr})dx"
    result = Function(antiderivative_expr)

    # Replace the __call__ method
    result.__call__ = antiderivative_func

    return result


def romberg_integration(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_steps: int = 10,
    tol: float = 1e-10,
) -> float:
    """
    Compute the definite integral using Romberg integration.

    :param f: Function to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param max_steps: Maximum number of steps
    :param tol: Error tolerance
    :return: Approximate value of the definite integral
    """
    # Initialize the Romberg table
    R = [[0] * (max_steps + 1) for _ in range(max_steps + 1)]

    # Compute the first term using the trapezoid rule with h = b-a
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_steps + 1):
        # Compute the trapezoid rule with step size h/2^i
        h = h / 2

        # Add the new points to the trapezoid sum
        sum_val = 0
        for k in range(1, 2 ** (i - 1) + 1):
            sum_val += f(a + (2 * k - 1) * h)

        # Compute the trapezoid approximation with step size h
        R[i][0] = 0.5 * R[i - 1][0] + h * sum_val

        # Apply Richardson extrapolation
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4**j - 1)

        # Check for convergence
        if i >= 2 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i]

    # Return the best approximation if max_steps is reached
    return R[max_steps][max_steps]
