"""
Basic usage examples for libfuncs.
"""
import numpy as np
import matplotlib.pyplot as plt
from libfuncs import Function

# Create and evaluate a simple function
f = Function("x^2 - 4")
print(f"f(2) = {f(2)}")  # Should be 0
print(f"f(3) = {f(3)}")  # Should be 5

# Find zeros
zeros = f.zeros()
print(f"Zeros of {f}: {zeros}")  # Should be [-2.0, 2.0]

# Compute derivative
df = f.derivative()
print(f"Derivative of {f}: {df}")
print(f"df(3) = {df(3)}")  # Should be 6

# Plot function
plt.figure(figsize=(10, 6))
f.plot(x_range=(-5, 5), show_plot=False)
plt.title("Function and its zeros")
plt.grid(True)
plt.scatter(zeros, [0] * len(zeros), color='red', s=100, label='Zeros')
plt.legend()
plt.savefig("function_zeros.png")
plt.close()
