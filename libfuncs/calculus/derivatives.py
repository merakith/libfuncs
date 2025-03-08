import numpy as np
from typing import Union, Callable, Dict, Any, Optional, List
from ..core.function import Function


def compute_derivative(
    func: Function, n: int = 1, method: str = "central", h: float = 1e-6
) -> Function:
    """
    Compute the nth derivative of a function.

    :param func: Function object to differentiate
    :param n: Order of the derivative (default: 1)
    :param method: Method to use ('central', 'forward', 'backward' or 'symbolic')
    :param h: Step size for numerical differentiation
    :return: New Function object representing the derivative
    :raises ValueError: If n < 1 or method is invalid
    """
    if n < 1:
        raise ValueError("Derivative order must be at least 1")

    if method not in ("central", "forward", "backward", "symbolic"):
        raise ValueError(f"Unknown differentiation method: {method}")

    # Try symbolic differentiation first if SymPy is available
    if method == "symbolic":
        try:
            return symbolic_derivative(func, n)
        except Exception as e:
            # Fall back to numerical if symbolic fails
            print(
                f"Symbolic differentiation failed: {e}. Falling back to central difference."
            )
            method = "central"

    # Create a new function that computes the derivative numerically
    def derivative_func(*args, **kwargs):
        if args and not kwargs:
            kwargs = {"x": args[0]}  # Convert positional to keyword args

        # Get the variable names
        variables = list(kwargs.keys())

        if not variables:
            raise ValueError("No variables provided for derivative evaluation")

        # For multivariate functions, we differentiate with respect to the first variable
        var_name = variables[0]
        var_value = kwargs[var_name]

        # Create a function of one variable for differentiation
        def f(t):
            kwargs_copy = kwargs.copy()
            kwargs_copy[var_name] = t
            return func(**kwargs_copy)

        # Compute the derivative at the point
        if n == 1:
            return numerical_derivative(f, var_value, method, h)
        else:
            # For higher-order derivatives, recursively apply the derivative
            def g(t):
                kwargs_copy = kwargs.copy()
                kwargs_copy[var_name] = t
                return numerical_derivative(f, t, method, h)

            return compute_derivative_recursive(g, var_value, n - 1, method, h)

    # Generate an expression string for the derivative that's safe for evaluation
    if n == 1:
        derivative_expr = f"2*x"  # Just a placeholder that won't cause parsing errors
    else:
        derivative_expr = f"{n}*x"  # Another placeholder
    
    # Create and return a new Function object
    result = Function(derivative_expr)
    
    # Replace the __call__ method with our numerical derivative
    result.__call__ = derivative_func
    
    # Store the original expression for display purposes
    if hasattr(result, '_original_expr'):
        if n == 1:
            result._original_expr = f"d/dx({func.expr})"
        else:
            result._original_expr = f"d{n}/dx{n}({func.expr})"
    
    return result


def numerical_derivative(
    f: Callable, x: float, method: str = "central", h: float = 1e-6
) -> float:
    """
    Compute the numerical derivative of a function at a point.

    :param f: Function to differentiate
    :param x: Point at which to compute derivative
    :param method: Differentiation method ('central', 'forward', or 'backward')
    :param h: Step size
    :return: Numerical derivative value
    """
    if method == "central":
        # Central difference formula: (f(x+h) - f(x-h)) / (2*h)
        return (f(x + h) - f(x - h)) / (2 * h)

    elif method == "forward":
        # Forward difference formula: (f(x+h) - f(x)) / h
        return (f(x + h) - f(x)) / h

    elif method == "backward":
        # Backward difference formula: (f(x) - f(x-h)) / h
        return (f(x) - f(x - h)) / h

    else:
        raise ValueError(f"Unknown differentiation method: {method}")


def compute_derivative_recursive(
    f: Callable, x: float, n: int, method: str = "central", h: float = 1e-6
) -> float:
    """
    Recursively compute higher-order derivatives.

    :param f: Function to differentiate
    :param x: Point at which to compute derivative
    :param n: Order of the derivative
    :param method: Differentiation method
    :param h: Step size
    :return: Value of the nth derivative at x
    """
    if n == 0:
        return f(x)

    if n == 1:
        return numerical_derivative(f, x, method, h)

    # Define the first derivative as a new function
    def df(t):
        return numerical_derivative(f, t, method, h)

    # Recursively compute (n-1)th derivative of df
    return compute_derivative_recursive(df, x, n - 1, method, h)


def symbolic_derivative(func: Function, n: int = 1) -> Function:
    """
    Compute symbolic derivative using SymPy.

    :param func: Function object to differentiate
    :param n: Order of the derivative
    :return: New Function object representing the symbolic derivative
    :raises ImportError: If SymPy is not available
    """
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("SymPy must be installed for symbolic differentiation")

    # Extract variable names using a simple parser
    from ..core.parser import extract_variables

    var_names = extract_variables(func.expr)

    if not var_names:
        var_names = ["x"]  # Default to 'x' if no variables found

    # Create symbolic variables for each variable in the expression
    sym_vars = {name: sp.Symbol(name) for name in var_names}

    # Try to convert the function expression to a SymPy expression
    # First replace some common mathematical operations
    expr = func.expr

    try:
        # Convert to a SymPy expression
        sym_expr = sp.sympify(expr)

        # Calculate the nth derivative with respect to the first variable
        deriv_var = sym_vars[var_names[0]]
        for _ in range(n):
            sym_expr = sp.diff(sym_expr, deriv_var)

        # Convert back to a string
        derivative_expr = str(sym_expr)

        # Create a new Function object with the derivative expression
        return Function(derivative_expr)

    except Exception as e:
        raise ValueError(f"Failed to compute symbolic derivative: {e}")


def partial_derivative(func: Function, var_name: str, n: int = 1) -> Function:
    """
    Compute the partial derivative with respect to a specific variable.

    :param func: Function object to differentiate
    :param var_name: Variable name to differentiate with respect to
    :param n: Order of the derivative
    :return: New Function object representing the partial derivative
    """

    # Create a wrapper function for the numerical computation
    def partial_derivative_func(*args, **kwargs):
        if args and not kwargs:
            # For partial derivatives we need named arguments
            raise ValueError("Partial derivatives require named arguments")

        # Check if variable exists in kwargs
        if var_name not in kwargs:
            raise ValueError(f"Variable {var_name} not provided in arguments")

        var_value = kwargs[var_name]

        # Create a function of one variable (the variable we're differentiating with respect to)
        def f(t):
            kwargs_copy = kwargs.copy()
            kwargs_copy[var_name] = t
            return func(**kwargs_copy)

        # Compute the derivative at the point
        if n == 1:
            return numerical_derivative(f, var_value)
        else:
            return compute_derivative_recursive(f, var_value, n)

    # Create a new Function with modified __call__ method
    derivative_expr = f"∂/∂{var_name}({func.expr})"
    result = Function(derivative_expr)
    result.__call__ = partial_derivative_func

    return result


def gradient(func: Function) -> Dict[str, Function]:
    """
    Compute the gradient of a multivariate function.

    :param func: Function object
    :return: Dictionary mapping variable names to their partial derivatives
    """
    from ..core.parser import extract_variables

    variables = extract_variables(func.expr)

    gradient_dict = {}
    for var in variables:
        gradient_dict[var] = partial_derivative(func, var)

    return gradient_dict


def directional_derivative(
    func: Function, point: Dict[str, float], direction: Dict[str, float]
) -> float:
    """
    Compute the directional derivative at a point.

    :param func: Function object
    :param point: Point at which to compute the derivative
    :param direction: Direction vector
    :return: Value of directional derivative
    """
    # Compute the gradient at the point
    grad = gradient(func)

    # Compute the dot product of gradient and direction
    dot_product = 0.0
    for var, deriv in grad.items():
        if var in point and var in direction:
            dot_product += deriv(**point) * direction[var]

    # Normalize the direction vector
    norm = np.sqrt(sum(d**2 for d in direction.values()))
    if norm != 0:
        return dot_product / norm
    else:
        return 0.0
