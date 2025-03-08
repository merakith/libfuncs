"""Benchmarks for root-finding methods."""

import time
import numpy as np
from libfuncs import Function
from libfuncs.analysis.zeros import find_zeros, find_zeros_newton

# Test functions with known roots
TEST_FUNCTIONS = [
    {"expr": "x^2 - 4", "name": "Quadratic", "roots": [-2, 2]},
    {"expr": "x^3 - x", "name": "Cubic", "roots": [-1, 0, 1]},
    {"expr": "sin(x)", "name": "Sine", "search_range": (-10, 370)},
    {"expr": "e^x - 5", "name": "Exponential", "roots": [1.6094]},
    {"expr": "x*log(abs(x))", "name": "x*log(x)", "roots": [0, 1]},
]

def benchmark_method(method_func, func, search_range=(-10, 10), **kwargs):
    """Benchmark a root-finding method."""
    start_time = time.time()
    roots = method_func(func, search_range, **kwargs)
    elapsed = time.time() - start_time
    return roots, elapsed

def run_benchmarks():
    """Run benchmarks for different root-finding methods."""
    print(f"{'Function':<15} {'Method':<15} {'Time (ms)':<10} {'Roots Found':<15}")
    print("-" * 60)
    
    for test_case in TEST_FUNCTIONS:
        expr = test_case["expr"]
        name = test_case["name"]
        search_range = test_case.get("search_range", (-10, 10))
        
        # Create function
        f = Function(expr)
        
        # Benchmark bisection method
        bisection_roots, bisection_time = benchmark_method(
            find_zeros, f, search_range=search_range
        )
        
        # Benchmark Newton's method
        newton_roots, newton_time = benchmark_method(
            find_zeros_newton, f, search_range=search_range
        )
        
        # Report results
        print(f"{name:<15} {'Bisection':<15} {bisection_time*1000:<10.2f} {len(bisection_roots)}")
        print(f"{'':<15} {'Newton':<15} {newton_time*1000:<10.2f} {len(newton_roots)}")

if __name__ == "__main__":
    run_benchmarks()
