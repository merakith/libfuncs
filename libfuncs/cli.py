"""Command-line interface for libfuncs."""

import argparse
import numpy as np
from .core.function import Function

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="libfuncs: Mathematical function analysis")
    parser.add_argument('expression', help="Mathematical expression to analyze")
    parser.add_argument('--eval', '-e', type=float, help="Evaluate function at this value")
    parser.add_argument('--zeros', '-z', action='store_true', help="Find zeros of the function")
    parser.add_argument('--derivative', '-d', action='store_true', help="Show derivative")
    parser.add_argument('--integral', '-i', nargs=2, type=float, metavar=('A', 'B'), 
                        help="Compute integral from A to B")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the function")
    
    args = parser.parse_args()
    
    # Create function
    try:
        f = Function(args.expression)
        print(f"Function: f(x) = {args.expression}")
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return
    
    # Process commands
    if args.eval is not None:
        x = args.eval
        print(f"f({x}) = {f(x)}")
    
    if args.zeros:
        zeros = f.zeros()
        print(f"Zeros: {zeros}")
    
    if args.derivative:
        df = f.derivative()
        print(f"Derivative: {df}")
    
    if args.integral:
        a, b = args.integral
        integral = f.integral(a, b)
        print(f"∫({a},{b}) f(x) dx = {integral}")
    
    if args.plot:
        import matplotlib.pyplot as plt
        f.plot()
        plt.show()

if __name__ == "__main__":
    main()
