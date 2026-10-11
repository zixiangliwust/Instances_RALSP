"""Example scripts for Simple Assembly Line Balancing Problem Type 1 (SALBP1).

This package contains example scripts demonstrating how to use various
optimization algorithms to solve assembly line balancing problems:

- SALBP1 examples: Simple Assembly Line Balancing Problem Type 1 with different
  solution representations (float, permutation) and mathematical models
- UALBP1 examples: U-shaped Assembly Line Balancing Problem Type 1 variants
- Heuristic examples: Simple heuristic methods for quick solutions

These examples show how to configure algorithms with appropriate operators for
assembly line balancing problems, where tasks must be assigned to workstations
while respecting precedence constraints and cycle time limits.
"""
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

