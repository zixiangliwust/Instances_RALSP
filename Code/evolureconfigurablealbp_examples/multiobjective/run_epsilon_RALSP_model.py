"""
Script to run the epsilon-constraint method and generate a set of Pareto frontier solutions
for the Multi-Objective Reconfigurable Assembly Line Balancing Problem.

This script can run on a single instance or all instances in the specified directory.
Results for each instance are saved in the 'results/' subfolder.
"""

import sys
import os
import winsound
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Add CPLEX path to sys.path only if it's not already there
cplex_path = r'D:/Cplex/cplex/python/3.10/x64_win64'
if cplex_path not in sys.path:
    sys.path.insert(0, cplex_path)

import math
import numpy as np
import time
import csv
from typing import List, Dict, Tuple

from evolureconfigurablealbp.problem.multiobjective.ReconfigurableALS_model import ReconfigurableALSPModel
from evolureconfigurablealbp.problem.multiobjective.MO_ReconfigurableALS_problem import IntegerMOReconfigurableALSP

def run_epsilon_constraint_pareto_frontier(file_name: str,
                                           file_path: str = "D:/GitHubInstances/Instances_RALSP/Instances/",
                                           max_seconds: int = 300,
                                           steps: int = 5) -> Tuple[List[Dict], List[Dict], int, float, int]:
    """
    Run the epsilon-constraint method on a single instance.

    Args:
        file_name: Name of the problem instance file.
        file_path: Directory containing the instance file.
        max_seconds: Maximum solving time per MILP model (seconds).
        steps: Number of steps to divide each objective's range.
    Returns:
        Tuple (all_solutions, pareto_solutions, total_combinations, total_runtime_seconds, num_optimal)
    """
    print("Generating Pareto Frontier Solutions using Epsilon-Constraint Method")
    print("=" * 80)
    print(f"Instance: {file_name}")
    print(f"Steps per objective: {steps} (-> {steps+1} finite epsilon values + inf)")
    print(f"Max seconds per model: {max_seconds}")
    print("=" * 80)

    print(f"Loading problem data from: {file_path}{file_name}")
    problem = IntegerMOReconfigurableALSP(file_path=file_path, file_name=file_name)
    print("Problem loaded successfully!")
    print(f"Number of products: {problem.number_of_product_types}")
    print(f"Number of workstations: {problem.number_of_workstations}")
    print(f"Number of parts: {problem.number_of_parts}")
    print(f"Product cycles: {problem.product_cycle}")

    # Create the model solver
    model_solver = ReconfigurableALSPModel(problem, max_seconds=max_seconds)

    # Step 1: Get min/max values for each objective
    print("\nFinding min/max values for each objective...")
    obj_ranges = {
        'model1': {'min': float('inf'), 'max': -float('inf')},
        'model2': {'min': float('inf'), 'max': -float('inf')},
        'model3': {'min': float('inf'), 'max': -float('inf')}
    }

    for obj in ['model1', 'model2', 'model3']:
        print(f"  Solving {obj} individually...")
        if obj == 'model1':
            res = model_solver.solve_model1()
        elif obj == 'model2':
            res = model_solver.solve_model2()
        else:  # model3
            res = model_solver.solve_model3()

        if res.get('solution') is not None:
            decoded = res.get('decoded_objectives', {})
            for k in obj_ranges:
                val = decoded.get(k, float('inf'))
                if val != float('inf'):
                    obj_ranges[k]['min'] = min(obj_ranges[k]['min'], val)
                    obj_ranges[k]['max'] = max(obj_ranges[k]['max'], val)
        else:
            # Fallback: use objective value if decoded missing
            obj_val = res.get('objective', float('inf'))
            if obj_val != float('inf'):
                obj_ranges[obj]['min'] = min(obj_ranges[obj]['min'], obj_val)
                obj_ranges[obj]['max'] = max(obj_ranges[obj]['max'], obj_val)

    # Ensure ranges are valid
    for k in obj_ranges:
        if obj_ranges[k]['min'] == float('inf'):
            obj_ranges[k]['min'] = 0.0
        if obj_ranges[k]['max'] == -float('inf'):
            obj_ranges[k]['max'] = obj_ranges[k]['min'] + 100.0

    print("Objectives ranges determined:")
    for obj, rng in obj_ranges.items():
        print(f"  {obj}: {rng['min']:.2f} - {rng['max']:.2f}")

    # Step 2: Generate epsilon combinations
    epsilon_combinations = []

    for primary in ['model1', 'model2', 'model3']:
        other1, other2 = [o for o in ['model1', 'model2', 'model3'] if o != primary]
        # Generate epsilon values (including min and max)
        eps1_vals = np.linspace(obj_ranges[other1]['min'], obj_ranges[other1]['max'], steps+1).tolist()
        eps2_vals = np.linspace(obj_ranges[other2]['min'], obj_ranges[other2]['max'], steps+1).tolist()
        # Add also a relaxed case (inf)
        for eps1 in [float('inf')] + eps1_vals:
            for eps2 in [float('inf')] + eps2_vals:
                epsilon_combinations.append({
                    'primary_objective': primary,
                    'epsilon_values': {other1: eps1, other2: eps2}
                })

    # Remove duplicates
    epsilon_combinations = list({(c['primary_objective'], frozenset(c['epsilon_values'].items())): c
                                for c in epsilon_combinations}.values())
    print(f"\nTotal epsilon combinations generated: {len(epsilon_combinations)}")

    # Step 3: Run each combination
    solutions = []
    all_records = []
    total_runtime = 0.0
    num_optimal = 0

    for idx, combo in enumerate(epsilon_combinations):
        print(f"\nRunning {idx+1}/{len(epsilon_combinations)}: Primary={combo['primary_objective']}, eps={combo['epsilon_values']}")

        # Prepare record for CSV
        record = {
            'primary_objective': combo['primary_objective'],
            'epsilon_values': combo['epsilon_values'],
            'status': '',
            'reconfiguration_cost': '',
            'part_frequency_violations': '',
            'logistics_leveling': '',
            'solve_time_sec': '',
            'product_sequence_first10': ''
        }

        start_time = time.time()
        try:
            res = model_solver.solve_epsilon_constraint_method(
                primary_objective=combo['primary_objective'],
                epsilon_values=combo['epsilon_values']
            )
            solve_time = time.time() - start_time
            record['solve_time_sec'] = f"{solve_time:.2f}"
            total_runtime += solve_time

            # 判断是否获得最优解（status 为 'optimal'）
            if 'optimal' in res.get('status'):
                num_optimal += 1

            if res['status'] != 'error' and res.get('solution'):
                decoded = res.get('decoded_objectives', {})
                reconf = decoded.get('reconfiguration_cost', float('inf'))
                viol = decoded.get('part_frequency_violations', float('inf'))
                level = decoded.get('logistics_leveling', float('inf'))

                if all(v != float('inf') for v in [reconf, viol, level]):
                    record['status'] = 'Success'
                    record['reconfiguration_cost'] = f"{reconf:.6f}"
                    record['part_frequency_violations'] = f"{viol:.6f}"
                    record['logistics_leveling'] = f"{level:.6f}"
                    seq = res.get('product_sequence', [])
                    record['product_sequence_first10'] = ','.join(map(str, seq[:10]))

                    sol = {
                        'primary_objective': combo['primary_objective'],
                        'epsilon_values': combo['epsilon_values'],
                        'reconfiguration_cost': reconf,
                        'part_frequency_violations': viol,
                        'logistics_leveling': level,
                        'product_sequence': seq,
                        'solve_time': solve_time
                    }
                    solutions.append(sol)
                    print(f"  Success: reconf={reconf:.2f}, viol={viol:.2f}, level={level:.2f}")
                else:
                    record['status'] = 'Incomplete objectives'
                    print("  Incomplete objectives, skipping")
            else:
                record['status'] = f"Failed: {res['status']}"
                print(f"  Failed - Status: {res['status']}")
        except Exception as e:
            solve_time = time.time() - start_time
            record['solve_time_sec'] = f"{solve_time:.2f}"
            total_runtime += solve_time
            record['status'] = f"Exception: {str(e)}"
            print(f"  Exception: {e}")

        all_records.append(record)

    # Step 4: Identify Pareto frontier
    pareto = identify_pareto_frontier(solutions)
    print(f"\nGenerated {len(solutions)} valid solutions, {len(pareto)} Pareto-optimal")
    print(f"Optimal solutions (status 'optimal'): {num_optimal} out of {len(epsilon_combinations)}")

    # Step 5: Write detailed records to CSV (instance-specific)
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_name = os.path.splitext(file_name)[0]
    detail_csv = os.path.join(results_dir, f"all_solutions_by_models_{base_name}.csv")
    with open(detail_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PrimaryObjective', 'Epsilon1Name', 'Epsilon1Value', 'Epsilon2Name', 'Epsilon2Value',
                         'Status', 'ReconfCost', 'Violations', 'LogisticsLeveling', 'SolveTimeSec', 'ProductSequenceFirst10'])
        for rec in all_records:
            eps_items = list(rec['epsilon_values'].items())
            if len(eps_items) >= 2:
                eps1_name, eps1_val = eps_items[0]
                eps2_name, eps2_val = eps_items[1]
            else:
                eps1_name = eps2_name = ''
                eps1_val = eps2_val = ''
            writer.writerow([
                rec['primary_objective'],
                eps1_name, eps1_val,
                eps2_name, eps2_val,
                rec['status'],
                rec['reconfiguration_cost'],
                rec['part_frequency_violations'],
                rec['logistics_leveling'],
                rec['solve_time_sec'],
                rec['product_sequence_first10']
            ])
    print(f"Detailed results saved to {detail_csv}")

    # Output Pareto frontier to file
    pareto_txt = os.path.join(results_dir, f"pareto_solutions_by_models_{base_name}.txt")
    with open(pareto_txt, 'w') as f:
        f.write("ProductSequence;ReconfCost;Violations;LogisticsLeveling\n")
        for sol in pareto:
            seq = ','.join(map(str, sol['product_sequence']))
            f.write(f"{seq};{sol['reconfiguration_cost']:.6f};{sol['part_frequency_violations']:.6f};{sol['logistics_leveling']:.6f}\n")
    print(f"Pareto solutions saved to {pareto_txt}")

    # Visualize
    try:
        pass
        # plot_pareto_frontier(solutions, pareto, base_name)
    except ImportError:
        print("Matplotlib not available - skipping visualization")

    return solutions, pareto, len(epsilon_combinations), total_runtime, num_optimal


def identify_pareto_frontier(solutions: List[Dict]) -> List[Dict]:
    """Identify the Pareto frontier from a set of solutions."""
    if not solutions:
        return []
    pareto = []
    for sol in solutions:
        dominated = False
        for other in solutions:
            if (other['reconfiguration_cost'] <= sol['reconfiguration_cost'] and
                other['part_frequency_violations'] <= sol['part_frequency_violations'] and
                other['logistics_leveling'] <= sol['logistics_leveling'] and
                (other['reconfiguration_cost'] < sol['reconfiguration_cost'] or
                 other['part_frequency_violations'] < sol['part_frequency_violations'] or
                 other['logistics_leveling'] < sol['logistics_leveling'])):
                dominated = True
                break
        if not dominated:
            duplicate = False
            for p in pareto:
                if (abs(p['reconfiguration_cost'] - sol['reconfiguration_cost']) < 1e-6 and
                    abs(p['part_frequency_violations'] - sol['part_frequency_violations']) < 1e-6 and
                    abs(p['logistics_leveling'] - sol['logistics_leveling']) < 1e-6):
                    duplicate = True
                    break
            if not duplicate:
                pareto.append(sol)
    return pareto


def plot_pareto_frontier(all_solutions: List[Dict], pareto_solutions: List[Dict], instance_name: str = "instance"):
    """Plot the Pareto frontier and save to file."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    all_reconf = [s['reconfiguration_cost'] for s in all_solutions]
    all_viol = [s['part_frequency_violations'] for s in all_solutions]
    all_level = [s['logistics_leveling'] for s in all_solutions]
    pareto_reconf = [s['reconfiguration_cost'] for s in pareto_solutions]
    pareto_viol = [s['part_frequency_violations'] for s in pareto_solutions]
    pareto_level = [s['logistics_leveling'] for s in pareto_solutions]

    if not all_reconf or not all_viol or not all_level:
        print("Could not generate plots: no valid solutions found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Pareto Frontier Analysis - {instance_name}', fontsize=16)

    axes[0, 0].scatter(all_viol, all_reconf, alpha=0.5, label='All Solutions', color='lightblue')
    axes[0, 0].scatter(pareto_viol, pareto_reconf, alpha=0.8, label='Pareto Solutions', color='red', s=100)
    axes[0, 0].set_xlabel('Part Frequency Violations')
    axes[0, 0].set_ylabel('Reconfiguration Cost')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(all_level, all_reconf, alpha=0.5, label='All Solutions', color='lightblue')
    axes[0, 1].scatter(pareto_level, pareto_reconf, alpha=0.8, label='Pareto Solutions', color='red', s=100)
    axes[0, 1].set_xlabel('Logistics Leveling')
    axes[0, 1].set_ylabel('Reconfiguration Cost')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(all_level, all_viol, alpha=0.5, label='All Solutions', color='lightblue')
    axes[1, 0].scatter(pareto_level, pareto_viol, alpha=0.8, label='Pareto Solutions', color='red', s=100)
    axes[1, 0].set_xlabel('Logistics Leveling')
    axes[1, 0].set_ylabel('Part Frequency Violations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.scatter(all_viol, all_reconf, all_level, alpha=0.3, label='All Solutions', color='lightblue')
    ax3d.scatter(pareto_viol, pareto_reconf, pareto_level, alpha=0.8, label='Pareto Solutions', color='red', s=100)
    ax3d.set_xlabel('Part Frequency Violations')
    ax3d.set_ylabel('Reconfiguration Cost')
    ax3d.set_zlabel('Logistics Leveling')

    plt.tight_layout()
    plt.savefig(f"results/pareto_frontier_{instance_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Pareto frontier plot saved as results/pareto_frontier_{instance_name}.png")


if __name__ == "__main__":
    # Path to instances directory
    file_path = "D:/GitHubInstances/Instances_RALSP/Instances/"
    # Get all files (instances) in the directory
    entries = os.listdir(file_path)
    problem_name_list = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    # Remove example_of_paper.txt from the list
    problem_name_list = [name for name in problem_name_list if name != "example_of_paper.txt" and name != "example_of_english_paper.txt"]
    problem_name_list = ["instance_001.txt", "instance_011.txt", "instance_020.txt",
                         "instance_021.txt", "instance_031.txt", "instance_040.txt",
                         "instance_041.txt", "instance_051.txt", "instance_060.txt",
                         "instance_061.txt", "instance_071.txt", "instance_080.txt",
                         "instance_081.txt", "instance_091.txt", "instance_100.txt",
                         "instance_101.txt", "instance_111.txt", "instance_120.txt"]

    # Summary data
    summary = []

    # Run each instance
    for problem_name in problem_name_list:
        print(f"\n\n{'='*80}")
        print(f"Processing instance: {problem_name}")
        print('='*80)

        solutions, pareto_solutions, num_combinations, total_runtime, num_optimal = run_epsilon_constraint_pareto_frontier(
            file_name=problem_name,
            file_path=file_path,
            max_seconds=300,
            steps=5
        )

        summary.append({
            'instance': problem_name,
            'total_combinations': num_combinations,
            'valid_solutions': len(solutions),
            'optimal_solutions': num_optimal,
            'pareto_solutions': len(pareto_solutions),
            'total_runtime_sec': total_runtime
        })

        print(f"\nInstance {problem_name} finished. Valid solutions: {len(solutions)}, Pareto: {len(pareto_solutions)}, Runtime: {total_runtime:.2f}s, Optimal: {num_optimal}")

    # Write summary to CSV
    summary_csv = "all_instances_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'TotalCombinations', 'ValidSolutions', 'OptimalSolutions', 'ParetoSolutions', 'TotalRuntimeSec'])
        for row in summary:
            writer.writerow([row['instance'], row['total_combinations'], row['valid_solutions'], row['optimal_solutions'], row['pareto_solutions'], f"{row['total_runtime_sec']:.2f}"])
    print(f"\nSummary saved to {summary_csv}")
    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
        print("End of beep.")
    except Exception:
        pass  # Ignore if winsound is not available on this platform