import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional, Union, Any
from ..core.function import Function
from ..core.evaluator import safe_evaluate, evaluate_over_range
from ..calculus.derivatives import compute_derivative
from ..calculus.extrema import find_extrema, find_inflection_points


def plot_function(
    func: Function,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Optional[Tuple[float, float]] = None,
    resolution: int = 1000,
    show_grid: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    color: str = "blue",
    line_style: str = "-",
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a function on the Cartesian plane.

    :param func: Function object to plot
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Optional tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points to plot
    :param show_grid: Whether to show a grid
    :param show_legend: Whether to show a legend
    :param title: Plot title (defaults to function expression)
    :param color: Line color
    :param line_style: Line style ('-', '--', ':', etc.)
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and axis objects with desired size
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x values within the specified range
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, resolution)

    # Compute y values, handling potential errors
    y_values = np.array([safe_evaluate(func, x) for x in x_values])

    # Filter out NaN and Inf values
    valid_indices = np.isfinite(y_values)
    x_valid = x_values[valid_indices]
    y_valid = y_values[valid_indices]

    # Plot the function
    ax.plot(
        x_valid,
        y_valid,
        color=color,
        linestyle=line_style,
        label=f"$f(x) = {func.expr}$",
    )

    # Set axis ranges
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add title and labels
    if title is None:
        title = f"Plot of $f(x) = {func.expr}$"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend if requested
    if show_legend:
        ax.legend()

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_derivative(
    func: Function,
    n: int = 1,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Optional[Tuple[float, float]] = None,
    resolution: int = 1000,
    show_original: bool = True,
    show_grid: bool = True,
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a function and its derivative.

    :param func: Function object to plot
    :param n: Order of derivative to plot
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Optional tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points to plot
    :param show_original: Whether to show the original function
    :param show_grid: Whether to show a grid
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and axis objects with desired size
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x values within the specified range
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, resolution)

    # Plot the original function if requested
    if show_original:
        y_values = np.array([safe_evaluate(func, x) for x in x_values])
        valid_indices = np.isfinite(y_values)
        ax.plot(
            x_values[valid_indices],
            y_values[valid_indices],
            color="blue",
            label=f"$f(x) = {func.expr}$",
        )

    # Compute the derivative
    deriv = compute_derivative(func, n=n)

    # Plot the derivative
    y_deriv_values = np.array([safe_evaluate(deriv, x) for x in x_values])
    valid_indices = np.isfinite(y_deriv_values)

    derivative_label = f"$f^{{{n}}}(x)$" if n > 1 else "$f'(x)$"
    ax.plot(
        x_values[valid_indices],
        y_deriv_values[valid_indices],
        color="red",
        label=derivative_label,
    )

    # Set axis ranges
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add title and labels
    suffix = ["st", "nd", "rd"] + ["th"] * 7
    derivative_ordinal = f"{n}{suffix[min(n-1, 10)]}" if n <= 10 else f"{n}th"
    title = f"Function and its {derivative_ordinal} Derivative"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.legend()

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_with_extrema(
    func: Function,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Optional[Tuple[float, float]] = None,
    resolution: int = 1000,
    show_inflection: bool = True,
    show_grid: bool = True,
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a function with its extrema and inflection points highlighted.

    :param func: Function object to plot
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Optional tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points to plot
    :param show_inflection: Whether to highlight inflection points
    :param show_grid: Whether to show a grid
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and axis objects with desired size
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x values within the specified range
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, resolution)

    # Compute y values, handling potential errors
    y_values = np.array([safe_evaluate(func, x) for x in x_values])
    valid_indices = np.isfinite(y_values)

    # Plot the function
    ax.plot(
        x_values[valid_indices],
        y_values[valid_indices],
        color="blue",
        label=f"$f(x) = {func.expr}$",
    )

    # Find extrema
    extrema_dict = find_extrema(func, x_range)

    # Plot maxima
    maxima = [x for x, type_val in extrema_dict.items() if type_val == "maxima"]
    if maxima:
        maxima_y = [func(x) for x in maxima]
        ax.scatter(
            maxima, maxima_y, color="red", s=100, marker="^", label="Maxima", zorder=5
        )

    # Plot minima
    minima = [x for x, type_val in extrema_dict.items() if type_val == "minima"]
    if minima:
        minima_y = [func(x) for x in minima]
        ax.scatter(
            minima, minima_y, color="green", s=100, marker="v", label="Minima", zorder=5
        )

    # Plot inflection points if requested
    if show_inflection:
        inflection_points = find_inflection_points(func, x_range)
        if inflection_points:
            inflection_y = [func(x) for x in inflection_points]
            ax.scatter(
                inflection_points,
                inflection_y,
                color="purple",
                s=100,
                marker="o",
                label="Inflection Points",
                zorder=5,
            )

    # Set axis ranges
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add title and labels
    ax.set_title(f"Function with Critical Points: $f(x) = {func.expr}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.legend()

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_integral(
    func: Function,
    a: float,
    b: float,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    resolution: int = 1000,
    show_grid: bool = True,
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a function and visualize the area under the curve (integral).

    :param func: Function object to integrate
    :param a: Lower limit of integration
    :param b: Upper limit of integration
    :param x_range: Tuple of (min_x, max_x) for the plot range (defaults to slightly wider than [a,b])
    :param y_range: Optional tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points to plot
    :param show_grid: Whether to show a grid
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and axis objects with desired size
    fig, ax = plt.subplots(figsize=figure_size)

    # Determine x range if not specified
    if x_range is None:
        margin = 0.2 * abs(b - a)
        x_range = (min(a, b) - margin, max(a, b) + margin)

    # Generate x values within the specified range
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, resolution)

    # Compute y values, handling potential errors
    y_values = np.array([safe_evaluate(func, x) for x in x_values])
    valid_indices = np.isfinite(y_values)

    # Plot the function
    ax.plot(
        x_values[valid_indices],
        y_values[valid_indices],
        color="blue",
        label=f"$f(x) = {func.expr}$",
    )

    # Generate x values for the integral region
    x_integral = np.linspace(a, b, resolution // 10)
    y_integral = np.array([safe_evaluate(func, x) for x in x_integral])

    # Fill the area under the curve
    ax.fill_between(
        x_integral,
        y_integral,
        np.zeros_like(y_integral),
        alpha=0.3,
        color="green",
        label=f"$\\int_{{{a}}}^{{{b}}} f(x) dx$",
    )

    # Add vertical lines at integration boundaries
    ax.axvline(x=a, color="red", linestyle="--", alpha=0.7, label=f"x = {a}")
    ax.axvline(x=b, color="red", linestyle="--", alpha=0.7, label=f"x = {b}")

    # Set axis ranges
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add title and labels
    ax.set_title(f"Integral of $f(x) = {func.expr}$ from {a} to {b}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.legend()

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_multiple_functions(
    functions: List[Function],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Optional[Tuple[float, float]] = None,
    resolution: int = 1000,
    show_grid: bool = True,
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple functions on the same graph.

    :param functions: List of Function objects to plot
    :param labels: Optional list of labels for each function
    :param colors: Optional list of colors for each function
    :param styles: Optional list of line styles for each function
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Optional tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points to plot
    :param show_grid: Whether to show a grid
    :param title: Plot title
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Check inputs
    n_funcs = len(functions)

    # Use default labels if not provided
    if labels is None:
        labels = [f"$f_{i+1}(x) = {func.expr}$" for i, func in enumerate(functions)]
    elif len(labels) < n_funcs:
        labels = labels + [f"$f_{i+1}(x)$" for i in range(len(labels), n_funcs)]

    # Use default colors if not provided
    if colors is None:
        # Use a colormap to generate colors
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, n_funcs - 1)) for i in range(n_funcs)]
    elif len(colors) < n_funcs:
        # Repeat colors if there aren't enough
        colors = colors * (n_funcs // len(colors) + 1)
        colors = colors[:n_funcs]

    # Use default styles if not provided
    if styles is None:
        styles = ["-"] * n_funcs
    elif len(styles) < n_funcs:
        # Repeat styles if there aren't enough
        styles = styles * (n_funcs // len(styles) + 1)
        styles = styles[:n_funcs]

    # Create figure and axis objects with desired size
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x values within the specified range
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, resolution)

    # Plot each function
    for i, func in enumerate(functions):
        # Compute y values, handling potential errors
        y_values = np.array([safe_evaluate(func, x) for x in x_values])
        valid_indices = np.isfinite(y_values)

        # Plot the function
        ax.plot(
            x_values[valid_indices],
            y_values[valid_indices],
            color=colors[i],
            linestyle=styles[i],
            label=labels[i],
        )

    # Set axis ranges
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add title and labels
    if title is None:
        title = "Multiple Functions" if n_funcs > 1 else f"Function: {labels[0]}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.legend()

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_3d_function(
    func: Function,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    z_range: Optional[Tuple[float, float]] = None,
    resolution: int = 100,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (10, 8),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 3D surface for a bivariate function f(x,y).

    :param func: Function object with two variables
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Tuple of (min_y, max_y) for the plot range
    :param z_range: Optional tuple of (min_z, max_z) for the plot range
    :param resolution: Number of points in each dimension
    :param colormap: Matplotlib colormap name
    :param title: Plot title
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")

    # Generate x and y grids
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute Z values
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = safe_evaluate(func, x=X[i, j], y=Y[i, j])

    # Replace any NaN or Inf values with zeros
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip Z values if z_range is provided
    if z_range:
        z_min, z_max = z_range
        Z = np.clip(Z, z_min, z_max)

    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z, cmap=plt.get_cmap(colormap), linewidth=0, antialiased=True, alpha=0.8
    )

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="f(x,y)")

    # Add title and labels
    if title is None:
        title = f"Surface Plot of $f(x,y) = {func.expr}$"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if z_range:
        ax.set_zlim(z_range)

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_contour(
    func: Function,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    num_contours: int = 20,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (10, 8),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a contour plot for a bivariate function f(x,y).

    :param func: Function object with two variables
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points in each dimension
    :param num_contours: Number of contour levels
    :param colormap: Matplotlib colormap name
    :param title: Plot title
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x and y grids
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute Z values
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = safe_evaluate(func, x=X[i, j], y=Y[i, j])

    # Replace any NaN or Inf values with zeros
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Create contour plot
    contour = ax.contourf(X, Y, Z, num_contours, cmap=plt.get_cmap(colormap))

    # Add contour lines with labels
    contour_lines = ax.contour(
        X, Y, Z, num_contours // 2, colors="black", linewidths=0.5
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%1.1f")

    # Add colorbar
    fig.colorbar(contour, label="f(x,y)")

    # Add title and labels
    if title is None:
        title = f"Contour Plot of $f(x,y) = {func.expr}$"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_vector_field(
    funcs: List[Function],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 20,
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (10, 8),
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 2D vector field.

    :param funcs: List of two Function objects [fx, fy] representing vector components
    :param x_range: Tuple of (min_x, max_x) for the plot range
    :param y_range: Tuple of (min_y, max_y) for the plot range
    :param resolution: Number of points in each dimension
    :param title: Plot title
    :param figure_size: Figure size in inches
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot image (if provided)
    :return: Matplotlib figure object
    """
    if len(funcs) != 2:
        raise ValueError(
            "For a vector field plot, exactly 2 functions must be provided"
        )

    fx, fy = funcs

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate x and y grids
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute vector components
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            U[i, j] = safe_evaluate(fx, x=X[i, j], y=Y[i, j])
            V[i, j] = safe_evaluate(fy, x=X[i, j], y=Y[i, j])

    # Replace any NaN or Inf values with zeros
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute vector field magnitude for color mapping
    magnitude = np.sqrt(U**2 + V**2)

    # Normalize vectors for clearer visualization
    norm = np.sqrt(U**2 + V**2)
    norm_factor = np.percentile(norm[norm > 0], 90)  # Use 90th percentile for scaling

    # Plot the vector field
    quiver = ax.quiver(
        X,
        Y,
        U,
        V,
        magnitude,
        cmap="viridis",
        pivot="mid",
        units="xy",
        scale=norm_factor * 2,
        scale_units="xy",
    )

    # Add colorbar
    fig.colorbar(quiver, label="Vector magnitude")

    # Add title and labels
    if title is None:
        title = "Vector Field Plot"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Display or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig
