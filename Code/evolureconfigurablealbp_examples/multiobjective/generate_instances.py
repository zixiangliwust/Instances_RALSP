"""
Generate 120 benchmark instances for Reconfigurable Assembly Line Scheduling Problem (RALSP).

Instance distribution:
- Station counts: 5, 10, 15, 20, 25, 30 (each generates 20 instances)
- Total: 6 × 20 = 120 instances

Demand patterns (minimum production cycles):
- 20 predefined patterns with total demand from 10 to 29
- Each pattern has GCD = 1 (no further reduction needed)
- For each station count, instances 1-20 use patterns 1-20 respectively

Other parameters are generated randomly with a fixed seed for reproducibility.

Usage:
    python generate_instances.py [--seed SEED] [--output DIR]

Example:
    python generate_instances.py --seed 42 --output D:/instances
"""

import os
import argparse
import random
from typing import List, Tuple

# ---------------------------- Configuration ----------------------------
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "D:/instances"

# Station counts (each with 20 instances)
STATION_COUNTS = [5, 10, 15, 20, 25, 30]
INSTANCES_PER_STATION_COUNT = 20

# Fixed parameters
NUM_PRODUCT_TYPES = 4
INTERVAL_TIME = 5

# Parameter ranges
PROCESSING_TIME_RANGE = (6, 25)    # per (product, station)
RECONFIG_COST_RANGE = (0, 12)      # off-diagonal cost (thousand CNY)
NUM_PARTS_RANGE = (12, 16)         # number of part types
PART_REQ_RANGE = (0, 8)            # integer, 0 means not required
STATION_LEN_RANGE = (5, 20)        # station length (unused but kept for format)
B_L_OPTIONS = [3, 4, 5]            # window length for part frequency constraints

# Pre-defined product demand patterns (minimum production cycles)
# Total demand from 10 to 29, each pattern has GCD = 1
DEMAND_PATTERNS = [
    [2, 2, 3, 3], # total 10
    [2, 2, 3, 4], # total 11
    [2, 3, 3, 4], # total 12
    [2, 3, 4, 4], # total 13
    [2, 3, 4, 5], # total 14
    [3, 3, 4, 5], # total 15
    [3, 4, 4, 5], # total 16
    [3, 4, 5, 5], # total 17
    [3, 4, 5, 6], # total 18
    [4, 4, 5, 6], # total 19
    [4, 5, 5, 6], # total 20
    [4, 5, 6, 6], # total 21
    [4, 5, 6, 7], # total 22
    [5, 5, 6, 7], # total 23
    [5, 6, 6, 7], # total 24
    [5, 6, 7, 7], # total 25
    [5, 6, 7, 8], # total 26
    [6, 6, 7, 8], # total 27
    [6, 7, 7, 8], # total 28
    [6, 7, 8, 8], # total 29
]
assert len(DEMAND_PATTERNS) == INSTANCES_PER_STATION_COUNT

# ---------------------------- Helper Functions ----------------------------
def random_processing_times(rng: random.Random, num_stations: int) -> List[List[int]]:
    """Processing times for each product at each station."""
    times = []
    for prod in range(1, NUM_PRODUCT_TYPES + 1):
        row = [prod] + [rng.randint(*PROCESSING_TIME_RANGE) for _ in range(num_stations)]
        times.append(row)
    return times

def random_reconfig_cost_matrix(rng: random.Random) -> List[Tuple[int, int, int]]:
    """Generate switching cost matrix entries (i, j, cost)."""
    entries = []
    for i in range(1, NUM_PRODUCT_TYPES + 1):
        for j in range(1, NUM_PRODUCT_TYPES + 1):
            if i == j:
                cost = 0
            else:
                cost = rng.randint(*RECONFIG_COST_RANGE)
            entries.append((i, j, cost))
    return entries

def random_part_frequency_constraints(rng: random.Random, num_parts: int) -> List[Tuple[int, int]]:
    """Generate (A_l, B_l) for each part."""
    constraints = []
    for _ in range(num_parts):
        B_l = rng.choice(B_L_OPTIONS)
        A_l = rng.randint(1, B_l)
        constraints.append((A_l, B_l))
    return constraints

def random_part_requirement_matrix(rng: random.Random, num_parts: int) -> List[List[int]]:
    """Generate part requirement matrix: each product has a list of integers (0-8)."""
    matrix = []
    for prod in range(1, NUM_PRODUCT_TYPES + 1):
        row = [rng.randint(*PART_REQ_RANGE) for _ in range(num_parts)]
        matrix.append(row)
    return matrix

def random_station_lengths(rng: random.Random, num_stations: int) -> List[int]:
    """Generate station lengths (unused but required by format)."""
    return [rng.randint(*STATION_LEN_RANGE) for _ in range(num_stations)]

# ---------------------------- File Writer ----------------------------
def write_instance(filepath: str, num_stations: int, demands: List[int],
                   station_lengths: List[int], processing_times: List[List[int]],
                   reconfig_entries: List[Tuple[int, int, int]],
                   num_parts: int, part_freq_constraints: List[Tuple[int, int]],
                   part_req_matrix: List[List[int]]):
    """Write a single instance file in the required format without extra blank lines."""
    with open(filepath, 'w') as f:
        # <number of products>
        f.write("<number of products>\n")
        f.write(f"{NUM_PRODUCT_TYPES}\n")
        
        # <product cycle>
        f.write("<product cycle>\n")
        for prod_idx, demand in enumerate(demands, start=1):
            f.write(f"{prod_idx} {demand}\n")
        
        # <number of workstations>
        f.write("<number of workstations>\n")
        f.write(f"{num_stations}\n")
        
        # <station length>
        f.write("<station length>\n")
        f.write(" ".join(map(str, station_lengths)) + "\n")
        
        # <working hours>
        f.write("<working hours>\n")
        for row in processing_times:
            f.write(" ".join(map(str, row)) + "\n")
        
        # <interval time>
        f.write("<interval time>\n")
        f.write(f"{INTERVAL_TIME}\n")
        
        # <product switching cost>
        f.write("<product switching cost>\n")
        for i, j, cost in reconfig_entries:
            f.write(f"{i},{j} {cost}\n")
        
        # <number of parts>
        f.write("<number of parts>\n")
        f.write(f"{num_parts}\n")
        
        # <optional frequency>
        f.write("<optional frequency>\n")
        for A_l, B_l in part_freq_constraints:
            f.write(f"{A_l} {B_l}\n")
        
        # <product assembly parts>
        f.write("<product assembly parts>\n")
        for row in part_req_matrix:
            f.write(" ".join(map(str, row)) + "\n")
        # No extra newline at the end

# ---------------------------- Instance Generation ----------------------------
def generate_instances_for_station_count(rng: random.Random, station_count: int,
                                         start_id: int, output_dir: str):
    """Generate INSTANCES_PER_STATION_COUNT instances with fixed station count."""
    for i in range(INSTANCES_PER_STATION_COUNT):
        instance_id = start_id + i
        filename = f"instance_{instance_id:03d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Use pre-defined demand pattern (index i)
        demands = DEMAND_PATTERNS[i]
        
        # Generate other parameters randomly using the RNG
        station_lengths = random_station_lengths(rng, station_count)
        processing_times = random_processing_times(rng, station_count)
        reconfig_entries = random_reconfig_cost_matrix(rng)
        num_parts = rng.randint(*NUM_PARTS_RANGE)
        part_freq_constraints = random_part_frequency_constraints(rng, num_parts)
        part_req_matrix = random_part_requirement_matrix(rng, num_parts)
        
        write_instance(filepath, station_count, demands, station_lengths,
                       processing_times, reconfig_entries, num_parts,
                       part_freq_constraints, part_req_matrix)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{INSTANCES_PER_STATION_COUNT} instances for stations={station_count}")

def main():
    parser = argparse.ArgumentParser(description="Generate RALSP benchmark instances")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print(f"Generating instances in directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Interval time: {INTERVAL_TIME}")
    print(f"Demand patterns: {INSTANCES_PER_STATION_COUNT} patterns, total demand from 10 to 29, GCD=1")
    print()
    
    next_id = 1
    for station_count in STATION_COUNTS:
        print(f"Stations = {station_count} -> {INSTANCES_PER_STATION_COUNT} instances")
        generate_instances_for_station_count(rng, station_count, next_id, args.output)
        next_id += INSTANCES_PER_STATION_COUNT
    
    total_instances = next_id - 1
    print(f"\nGeneration complete. {total_instances} instance files created.")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print("All instances are deterministic for the given seed.")

if __name__ == "__main__":
    main()