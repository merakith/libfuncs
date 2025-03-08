import pytest
import numpy as np
import matplotlib.pyplot as plt
from libfuncs import Function
from libfuncs.visualization.plotting import (
    plot_function, 
    plot_derivative,
    plot_with_extrema,
    plot_multiple_functions
)

@pytest.fixture
def disable_show(monkeypatch):
    """Fixture to prevent plt.show() from actually showing plots during tests"""
    monkeypatch.setattr(plt, 'show', lambda: None)

def test_plot_function(disable_show):
    """Test basic function plotting"""
    f = Function("x^2")
    fig = plot_function(f, show_plot=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_with_custom_range(disable_show):
    """Test plotting with custom range"""
    f = Function("sin(x)")
    fig = plot_function(f, x_range=(-180, 180), y_range=(-1.5, 1.5), show_plot=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_derivative(disable_show):
    """Test derivative plotting"""
    f = Function("x^3 - 2*x")
    fig = plot_derivative(f, n=1, show_plot=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_with_extrema(disable_show):
    """Test plotting with extrema points highlighted"""
    f = Function("x^3 - 3*x")
    fig = plot_with_extrema(f, show_plot=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_multiple_functions(disable_show):
    """Test plotting multiple functions"""
    f1 = Function("x^2")
    f2 = Function("x^3")
    f3 = Function("sin(x)")
    
    fig = plot_multiple_functions(
        [f1, f2, f3], 
        labels=["Quadratic", "Cubic", "Sine"],
        x_range=(-2, 2),
        show_plot=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
