# Libfuncs - Mathematical Function Library

Libfuncs is a Python library for mathematical function analysis, calculus operations, and visualization. It provides intuitive ways to work with mathematical functions and perform various operations on them.

## Features

- **Function representation**: Create and manipulate mathematical functions
- **Calculus operations**: Compute derivatives, integrals, find extrema points
- **Analysis**: Find zeros, poles, and roots of functions
- **Visualization**: Plot functions and their properties with matplotlib

## Installation

### Using pip

```bash
pip install libfuncs
```

### From source

```bash
# Clone the repository
git clone https://github.com/yourgithub/libfuncs.git
cd libfuncs

# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

### Development setup

```bash
# Install with development dependencies
poetry install --with dev

# Run tests
pytest

# Run code formatting
black .
```

## Quick Start

```python
from libfuncs import Function
import numpy as np

# Create a function
f = Function("x^2 - 4")

# Evaluate at a point
f(2)  # Returns 0

# Find zeros
zeros = f.zeros()  # [-2.0, 2.0]

# Compute derivative
df = f.derivative()  # 2*x
df(3)  # Returns 6

# Compute integral
area = f.integral(0, 3)  # Integral from 0 to 3

# Plotting
f.plot(x_range=(-5, 5))
```

## Advanced Usage

### Multivariable Functions

```python
f = Function("x^2 + y^2")
f(x=1, y=2)  # Returns 5
```

### Calculus Operations

```python
from libfuncs import compute_derivative, compute_integral, find_extrema

# Second derivative
f = Function("x^3 - 3*x")
f_second = compute_derivative(f, n=2)  # 6*x

# Integration methods
f = Function("1/x")
result = compute_integral(f, 1, 2, method="adaptive_simpson")

# Find extrema
extrema = find_extrema(f, (-5, 5))
```

### Advanced Plotting

```python
from libfuncs.visualization.plotting import (
    plot_function, 
    plot_derivative,
    plot_with_extrema,
    plot_multiple_functions
)

f = Function("x^3 - 3*x")

# Plot function with its derivative
plot_derivative(f, n=1)

# Plot function with extrema highlighted
plot_with_extrema(f)

# Plot multiple functions
plot_multiple_functions(
    [Function("x^2"), Function("sin(x)")],
    labels=["Quadratic", "Sine"]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
