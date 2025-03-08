"""Core functionality for the libfuncs library."""

from .function import Function
from .parser import parse_expression, extract_variables, validate_expression
from .evaluator import safe_evaluate, evaluate_expression, evaluate_over_range

__all__ = [
    'Function',
    'parse_expression',
    'extract_variables',
    'validate_expression',
    'safe_evaluate',
    'evaluate_expression',
    'evaluate_over_range',
]
