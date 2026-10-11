"""Single-objective optimization algorithms.

This module provides implementations of various single-objective evolutionary
algorithms and metaheuristics:

- Evolutionary algorithms: Genetic Algorithm (GA), Evolution Strategy (ES)
- Swarm intelligence: Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO),
  Whale Optimization Algorithm (WOA), etc.
- Physics-based: Simulated Annealing (SA), Sine Cosine Algorithm (SCA)
- Human-based: Teaching-Learning-Based Optimization (TLBO), JAYA
- Biology-based: Artificial Bee Colony (ABC), Moth-Flame Optimization (MFO)
- Math-based: Differential Evolution (DE), various local search algorithms

All algorithms are designed to find optimal solutions for problems with
a single objective function.
"""
from .ES import ESBase
from .GA import GABase
from .LS import LSBase
from .SA import SABase

"""
Motivations
The reasons of building this project are listed as follows. 1) There are some mistakes in the available open source
projects, hence I would like to eliminate some mistakes by my team and the help of other researchers. I am hoping for
10 researchers to confirm the codes of algorithms. If you find some mistake or you think that the codes are correct,
please contact me by email and your name, affiliation and email would be included in this project. 2) I want make the
codes easily to be extended to solve kinds of optimization problem, including one-objective, multi-objective continuous
and discrete optimization algorithms. It would be easy for me and my team to include many more problems if I have an
open project at my hand. In short, we have four purposes:
Purpose 1: Share the knowledge and help the researchers to eliminate the unnecessary duplication;
Purpose 2: Standardize the procedures of the algorithms as there are many variants available;
Purpose 3: Encode and identify the state-of-the methodologies with the help and corrections from the researchers;
Purpose 4: Achieve the simple and powerful algorithms by eliminate the complex and unnecessary improvements;
Acknowledgement
The codes are modified from the open source project. Thanks for the contributions of these researchers. The links are listed as follows.
1. https://github.com/thieu1995/mealpy
2. https://github.com/7ossam81/EvoloPy
Expecting your participation:
1) This project wants 10 researchers with SCI papers to confirm the decoding for each algorithm;
2) It is expected that you develop new algorithms or re-check the encoding or simplify the encoding;
3) It is also expected that you extend the algorithms to new optimization problems;
The name, affiliation and email addressed will be presented to show your contributions. Meanwhile, any update will be informed to the researchers who help doing the programming or checking the programming.
Classification of the applied algorithms:
Based on the links as follows:
1. https://linkinghub.elsevier.com/retrieve/pii/S1877050920319724
2. https://github.com/thieu1995/mealpy
Algorithm Categories (Category):
Evolutionary-based: Idea from Darwin's law of natural selection, evolutionary computing;
Swarm-based: Idea from movement, interaction of birds, organization of social;
Physics-based: Idea from physics law such as Newton's law of universal gravitation, black hole, multiverse;
Human-based: Idea from human interaction such as queuing search, teaching learning;
Biology-based: Idea from biology creature (or microorganism);
System-based: Idea from eco-system, immune-system, network-system;
Math-based: Idea from mathematical form or mathematical law such as sin-cosin;
Music-based: Idea from music instrument;
Probabilistic-base: Probabilistic based algorithm;
Dummy: Non-sense algorithms and non-sense papers (code proofs);
Algorithms’ performances (Performance):
Good: working good with benchmark functions (convergence good);
Not good: not working good with benchmark functions (convergence not good, not balance the exploration and exploitation phase);
Number of parameters (Paras):
The number of parameters in the algorithm (Not counting the fixed parameters in the original paper);
Almost algorithms have 2 paras (number of iterations, population size) and plus some paras depend on each algorithm;
Difficulty level(difficulty):
It is determined based on 1) the number of parameters, 2) number of equations, 3) time spend for coding, 4) source lines of code.
Easy: A few paras, few equations, source lines very short
Medium: More equations than Easy level, source lines longer than Easy level
Hard: Lots of equations, source lines longer than medium level, the paper hard to read.
Hard* - Very hard: Lots of equations, source lines too long, the paper is very hard to read.
Suggested algorithm in bold:
Ten algorithms will be suggested under 1) good performance; 2) a few of parameters; 2) easy programming. For instance,
some algorithms have "good" performance and have only 2 parameters will be highlighted in bold.
"""
