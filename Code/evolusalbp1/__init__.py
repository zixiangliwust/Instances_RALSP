"""Simple Assembly Line Balancing Problem Type 1 (SALBP1) module.

This package provides implementations of optimization problems for simple
assembly line balancing type 1, where the goal is to minimize the number of
workstations given a fixed cycle time.

Key features:
- Simple Assembly Line Balancing Problem (SALBP1) variants
- U-shaped Assembly Line Balancing Problem (UALBP1) variants
- Multiple solution representations (float, permutation, integer)
- Mathematical programming models (MIP and CP models)
- Lower bounding techniques (LB1, LB2, LB3)
- Heuristic solution generation
- Specialized optimizers (ACO, RS, SA, TS)

The problem addresses the challenge of assigning tasks to workstations on an
assembly line while respecting precedence constraints and cycle time limits.
"""

