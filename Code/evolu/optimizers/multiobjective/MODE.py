from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.algorithm import DynamicAlgorithm
from evolu.core.problem import DynamicProblem, Problem
from evolu.core.solution import FloatSolution
from evolu.core.operator import Crossover, Selection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement, RemovalPolicyType
from evolu.util.comparator import Comparator, DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = List[S]
"""
Module: GDE3 (Generalized differential evolution algorithm)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class GDE3(MultiObjectiveSwarmRoot[S, R]):
    """
    GDE3 (Generalized differential evolution algorithm)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Kukkonen, S., and J. Lampinen. 2005. GDE3: The third evolution step of generalized differential evolution.
    Paper presented at the 2005 IEEE Congress on Evolutionary Computation, 2-5 Sept. 2005.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 dominance_comparator: Comparator = store.default_comparator,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(GDE3, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Generalized differential evolution"
        self.selection_operator = selection
        self.crossover_operator = crossover        
        self.dominance_comparator = dominance_comparator
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(self.dominance_comparator), CrowdingDistance())
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))


    def selection(self, population: List[S]):
        mating_pool = []
        for i in range(0,self.offspring_population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solution = self.selection_operator.execute(self.solutions)
            # If the selection operator returns a list, extend mating_pool with it
            if isinstance(selected_solution, list):
                mating_pool.extend(selected_solution)
            else:
                # Otherwise, it returns a single solution, add it directly
                mating_pool.append(selected_solution)
        return mating_pool

    def reproduction(self, population: List[S]) -> List[S]:
        number_of_parents = self.crossover_operator.get_number_of_parents()        
        offsprings = []
        for i in range(0, self.offspring_population_size):
            parents = []
            for j in range(number_of_parents):
                parents.append(population[i * number_of_parents + j])
            new_solution = self.crossover_operator.execute(self.solutions[i], parents)
            offsprings.append(new_solution)
        return offsprings
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class DynamicGDE3(GDE3, DynamicAlgorithm):
    def __init__(self,
                 problem: DynamicProblem,
                 population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 dominance_comparator: Comparator = DominanceWithConstraintsComparator(),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(DynamicGDE3, self).__init__(
            problem=problem,
            population_size=population_size,
            selection=selection,
            crossover=crossover,
            dominance_comparator=dominance_comparator,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
            termination_criterion=termination_criterion,
        )
        self.completed_iterations = 0

    def restart(self) -> None:
        self.solutions = self.evaluate(self.solutions)

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.evaluations += self.offspring_population_size

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data["TERMINATION_CRITERIA_IS_MET"] = True
            self.observable.notify_all(**observable_data)
            self.restart()
            self.init_progress()
            self.completed_iterations += 1
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
