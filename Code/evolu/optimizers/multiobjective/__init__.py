"""Multi-objective optimization algorithms.

This module provides implementations of multi-objective evolutionary algorithms:

- NSGA-II, NSGA-III: Non-dominated sorting genetic algorithms
- MOEA/D: Multi-objective evolutionary algorithm based on decomposition
- SPEA2: Strength Pareto Evolutionary Algorithm 2
- IBEA: Indicator-Based Evolutionary Algorithm
- GDE3: Generalized Differential Evolution 3
- MOPSO variants: OMOPSO, SMPSO, SMPSORP
- HYPE: Hypervolume-based algorithm
- MOCell: Multi-objective cellular genetic algorithm
- MORS: Multi-objective Riesz s-Energy based selection
- MOHH: Multi-objective Hyper-heuristic algorithms with DQN and Q-Learning
- Dynamic variants: Algorithms adapted for dynamic optimization problems

All algorithms are designed to find Pareto-optimal solutions for problems
with multiple conflicting objectives.
"""
from .MODE import GDE3, DynamicGDE3
from .HYPE import HYPE
from .IBEA import IBEA
from .MOCell import MOCell
from .MOEAD import MOEAD, MOEAD_DRA, MOEADIEpsilon
from .MOHH import DQNMOHH, QMOHH
from .MOGAII import NSGAII, DistributedNSGAII, DynamicNSGAII
from .MOPSO import OMOPSO, SMPSO, SMPSORP, DynamicSMPSO
from .MORS import MORS
from .SPEA2 import SPEA2