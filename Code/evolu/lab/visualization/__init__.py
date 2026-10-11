"""Visualization tools for evolutionary algorithms.

This module provides various visualization capabilities:

- Plot: Basic plotting utilities for Pareto fronts, trajectories, etc.
- InteractivePlot: Interactive visualizations for exploration
- StreamingPlot: Real-time plotting during algorithm execution
- plot_posterior: Plot posterior distributions for Bayesian analysis
- linechart: Line chart utilities for convergence and performance metrics

These tools help researchers and practitioners visualize algorithm behavior,
solution quality, and comparative performance.
"""
# Try to import CDplot, but don't fail if statistical_test module is not available

from .interactive import InteractivePlot
from .plotting import Plot
from .posterior import plot_posterior
from .streaming import StreamingPlot

__all__ = ["Plot", "InteractivePlot", "StreamingPlot", "plot_posterior"]
