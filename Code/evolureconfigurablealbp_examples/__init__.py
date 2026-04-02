"""Example scripts for Reconfigurable Assembly Line Balancing Problem (Reconfigurable ALBP).

This package contains example scripts demonstrating how to use various
optimization algorithms to solve reconfigurable assembly line balancing problems:

- Multi-objective examples: MOReconfigurableALBP problems with float and integer
  solution representations
- Experimental execution: Systematic experimental studies comparing multiple algorithms

These examples show how to configure algorithms with appropriate operators for
reconfigurable assembly line balancing problems, where multiple product types
need to be sequenced efficiently while optimizing switching costs, production
leveling, and workload balancing.
"""
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

