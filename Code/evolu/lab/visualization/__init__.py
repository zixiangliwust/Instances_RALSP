"""Visualization tools for evolutionary algorithms.

This module provides various visualization capabilities:

- Plot: Basic plotting utilities for Pareto fronts, trajectories, etc.
- InteractivePlot: Interactive visualizations for exploration
- StreamingPlot: Real-time plotting during algorithm execution
- plot_posterior: Plot posterior distributions for Bayesian analysis
- CDplot: Plot critical distance diagrams for statistical comparisons
- linechart: Line chart utilities for convergence and performance metrics

These tools help researchers and practitioners visualize algorithm behavior,
solution quality, and comparative performance.
"""
from evolu.lab.statistical_test.critical_distance import CDplot
from .interactive import InteractivePlot
from .plotting import Plot
from .posterior import plot_posterior
from .streaming import StreamingPlot

__all__ = ["Plot", "InteractivePlot", "StreamingPlot", "CDplot", "plot_posterior"]
