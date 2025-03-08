import re
import numpy as np
from typing import Union, Dict, Any, Tuple, List

def parse_expression(expr: str) -> object:
    """
    Parse a mathematical expression string into a compiled Python expression.
    
    :param expr: Mathematical expression as a string
    :return: Compiled code object
    :raises ValueError: If the expression cannot be parsed
    """
    try:
        converted_expr = convert_expression(expr)
        return compile(converted_expr, "<string>", "eval")
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr}': {str(e)}")

def convert_expression(expr: str) -> str:
    """
    Convert mathematical notation to Python code.
    
    :param expr: Mathematical expression as a string
    :return: Python expression as a string
    """
    expr = _replace_constants(expr)
    expr = _replace_trigonometric_functions(expr)
    expr = _replace_logarithmic_functions(expr)
    expr = _replace_exponential_notation(expr)
    expr = _replace_absolute_value(expr)
    
    return expr

def _replace_constants(expr: str) -> str:
    """Replace mathematical constants with their numerical values."""
    # Replace pi and e with their numerical values
    expr = re.sub(r'\bpi\b', str(np.pi), expr)
    expr = re.sub(r'\be\b', str(np.e), expr)
    return expr

def _replace_trigonometric_functions(expr: str) -> str:
    """Convert trigonometric function notation to numpy equivalents."""
    # Standard trigonometric functions
    expr = re.sub(r'sin\(([^)]+)\)', r'np.sin(np.radians(\1))', expr)
    expr = re.sub(r'cos\(([^)]+)\)', r'np.cos(np.radians(\1))', expr)
    expr = re.sub(r'tan\(([^)]+)\)', r'np.tan(np.radians(\1))', expr)
    
    # Reciprocal trigonometric functions
    expr = re.sub(r'sec\(([^)]+)\)', r'1/np.cos(np.radians(\1))', expr)
    expr = re.sub(r'cosec\(([^)]+)\)', r'1/np.sin(np.radians(\1))', expr)
    expr = re.sub(r'csc\(([^)]+)\)', r'1/np.sin(np.radians(\1))', expr)  # Alternative notation
    expr = re.sub(r'cot\(([^)]+)\)', r'1/np.tan(np.radians(\1))', expr)
    
    # Inverse trigonometric functions
    expr = re.sub(r'asin\(([^)]+)\)', r'np.degrees(np.arcsin(\1))', expr)
    expr = re.sub(r'arcsin\(([^)]+)\)', r'np.degrees(np.arcsin(\1))', expr)
    expr = re.sub(r'acos\(([^)]+)\)', r'np.degrees(np.arccos(\1))', expr)
    expr = re.sub(r'arccos\(([^)]+)\)', r'np.degrees(np.arccos(\1))', expr)
    expr = re.sub(r'atan\(([^)]+)\)', r'np.degrees(np.arctan(\1))', expr)
    expr = re.sub(r'arctan\(([^)]+)\)', r'np.degrees(np.arctan(\1))', expr)
    
    # Hyperbolic functions
    expr = re.sub(r'sinh\(([^)]+)\)', r'np.sinh(\1)', expr)
    expr = re.sub(r'cosh\(([^)]+)\)', r'np.cosh(\1)', expr)
    expr = re.sub(r'tanh\(([^)]+)\)', r'np.tanh(\1)', expr)
    
    return expr

def _replace_logarithmic_functions(expr: str) -> str:
    """Convert logarithmic function notation to numpy equivalents."""
    # Natural logarithm
    expr = re.sub(r'ln\(([^)]+)\)', r'np.log(\1)', expr)
    
    # Base-10 logarithm
    expr = re.sub(r'log\(([^)]+)\)', r'np.log10(\1)', expr)
    expr = re.sub(r'log10\(([^)]+)\)', r'np.log10(\1)', expr)
    
    # Base-2 logarithm
    expr = re.sub(r'log2\(([^)]+)\)', r'np.log2(\1)', expr)
    
    # Custom base logarithm: log_b(x) becomes log(x)/log(b)
    expr = re.sub(r'log_([^(]+)\(([^)]+)\)', r'np.log(\2)/np.log(\1)', expr)
    
    return expr

def _replace_exponential_notation(expr: str) -> str:
    """Convert exponential notation (caret) to Python's power operator."""
    # Replace ^ with ** for exponentiation
    expr = re.sub(r'\^', '**', expr)
    
    # Handle scientific notation like 2e3
    expr = re.sub(r'(\d+)e(\d+)', r'\1*10**\2', expr)
    
    return expr

def _replace_absolute_value(expr: str) -> str:
    """Convert absolute value notation |x| to np.fabs(x)."""
    # Simple case: |x| → np.fabs(x)
    expr = re.sub(r'\|([^|]+)\|', r'np.fabs(\1)', expr)
    
    return expr

def extract_variables(expr: str) -> List[str]:
    """
    Extract variable names from an expression.
    
    :param expr: Mathematical expression as a string
    :return: List of variable names
    """
    # Convert expression to handle numpy functions
    converted = convert_expression(expr)
    
    # Remove all standard function names and constants
    known_names = [
        'np', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 
        'pi', 'e', 'sqrt', 'abs', 'fabs', 'radians', 'degrees',
        'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh'
    ]
    
    # Remove all numeric literals
    converted = re.sub(r'\d+(\.\d*)?', '', converted)
    
    # Extract potential variable names (alphanumeric sequences)
    potential_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\b', converted))
    
    # Filter out known function and constant names
    variables = [var for var in potential_vars if var not in known_names]
    
    return sorted(variables)

def validate_expression(expr: str) -> Tuple[bool, str]:
    """
    Validate that an expression is well-formed.
    
    :param expr: Mathematical expression as a string
    :return: Tuple of (is_valid, error_message)
    """
    # Check for balanced parentheses
    if expr.count('(') != expr.count(')'):
        return False, "Unbalanced parentheses"
        
    # Check for balanced absolute value symbols
    if expr.count('|') % 2 != 0:
        return False, "Unbalanced absolute value symbols"
    
    # Try to convert and compile the expression
    try:
        converted = convert_expression(expr)
        compile(converted, "<string>", "eval")
        return True, "Expression is valid"
    except Exception as e:
        return False, f"Invalid expression: {str(e)}"