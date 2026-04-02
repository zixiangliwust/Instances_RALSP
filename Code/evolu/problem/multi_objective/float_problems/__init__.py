"""Multi-objective continuous (float) optimization problems.

This module provides continuous multi-objective benchmark problems with
real-valued decision variables:

- ZDT: Zitzler-Deb-Thiele test suite
- DTLZ: Deb-Thiele-Laumanns-Zitzler test suite
- UF: Unconstrained functions from CEC 2009
- LZ09: Li-Zhang test suite
- FDA: Dynamic test problems (FDA1-FDA5)
- RE: Real-World Engineering problems (RE1-RE9)
- RWA: Real-World Application problems (RWA1-RWA10)
- Constrained: C-DTLZ, C1-DTLZ, LIR-CMOP series

These problems are widely used in the multi-objective optimization community
for algorithm benchmarking and comparison.
"""
from .constrained import *
from .DTLZ import *
from .fda import *
from .lircmop import *
from .LZ09 import *
from .re import *
from .rwa import *
from .uf import *
from .unconstrained import *
from .ZDT import *
