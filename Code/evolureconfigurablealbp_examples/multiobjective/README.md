# ReconfigurableALBP Multi-Objective Examples

This directory contains examples for solving the Multi-Objective Reconfigurable Assembly Line Balancing Problem (MOReconfigurableALBP) using various algorithms.

## Available Examples

### MODQNHH - DQN-Driven Hyper-Heuristic

**File**: `run_MODQNHH_FloatMOReconfigurableALBP.py`

A novel algorithm that uses Deep Q-Network (DQN) reinforcement learning to adaptively select evolutionary mechanisms from multiple algorithms (MOPSO, MOTLBO, MOWOA).

**Features**:
- Learns which mechanisms work best in different search states
- Combines strengths of PSO, TLBO, and WOA algorithms
- Provides detailed statistics on mechanism usage and success
- Adapts strategy based on problem characteristics

**Usage**:
```bash
python run_MODQNHH_FloatMOReconfigurableALBP.py
```

**Configuration**:
- Population size: 100
- Maximum evaluations: 25,000
- Archive size: 100
- DQN learning rate: 0.001
- 6 evolutionary mechanisms

**Output**:
- Pareto front saved to `results/MODQNHH_ReconfigurableALBP_FUN.txt`
- Variables saved to `results/MODQNHH_ReconfigurableALBP_VAR.txt`
- Mechanism selection statistics printed to console

## Problem Description

The Multi-Objective Reconfigurable Assembly Line Balancing Problem optimizes:

1. **Switching Cost**: Minimize cost of switching between product types
2. **Production Leveling**: Balance production rates across products
3. **Constraint Violations**: Minimize violations of optional frequency constraints

**Problem Characteristics**:
- Multiple product types with different cycle times
- Workstation capacity constraints
- Product switching costs
- Optional part frequency constraints

## Quick Start

1. **Prepare Problem Instance**:
   - Place problem files in `resources/ReconfigurableALBP/`
   - Update `file_name` in the script if needed

2. **Run Algorithm**:
   ```bash
   cd python/evolureconfigurablealbp_examples/multiobjective
   python run_MODQNHH_FloatMOReconfigurableALBP.py
   ```

3. **View Results**:
   - Check console output for statistics
   - Find solution files in `results/` directory
   - Analyze Pareto front trade-offs

## Understanding the Output

### Console Output

```
================================================================================
Multi-Objective DQN Hyper-Heuristic for Reconfigurable ALBP
================================================================================
Problem loaded successfully!
Number of products: 3
Number of workstations: 5
Number of variables: 30
...

MODQNHH Mechanism Selection Statistics
============================================================
Mechanism            Usage      Success    Success Rate   
------------------------------------------------------------
PSO-Position         45         28         62.22%
TLBO-Teacher         41         25         60.98%
...
```

### Result Files

**FUN.txt** (Objectives):
```
1234.5  567.8  2.0
1156.3  589.4  3.0
...
```
Each row: [Switching Cost, Leveling, Violations]

**VAR.txt** (Variables):
```
0.234 0.567 0.891 0.123 ...
0.345 0.678 0.912 0.234 ...
```
Each row: Float encoding of product sequence

## Customization

### Modify Algorithm Parameters

Edit the script:
```python
algorithm = MODQNHH(
    problem=problem,
    population_size=100,      # Increase for more diversity
    max_evaluations=25000,    # Increase for better convergence
    dqn_learning_rate=0.001,  # Adjust learning speed
    target_update_frequency=10  # Adjust stability vs. speed
)
```

### Use Different Problem Instance

Update in script:
```python
file_name = "ReconfigurableALBP_instance_02.txt"  # Your file
```

## Mechanism Descriptions

1. **PSO Position**: Particle swarm velocity-based updates
2. **TLBO Teacher**: Learning from best solution (teacher)
3. **TLBO Learner**: Peer-to-peer learning between solutions
4. **WOA Encircle**: Exploitation around best solution
5. **WOA Spiral**: Spiral movement pattern
6. **WOA Search**: Exploration using random solutions

The DQN learns when to use each mechanism based on the search state.

## Interpreting Statistics

```
Mechanism            Usage      Success    Success Rate   
PSO-Position         45         28         62.22%
```

- **Usage**: Number of times mechanism was selected
- **Success**: Number of times it led to improvement
- **Success Rate**: Percentage of successful applications

Higher success rates indicate more effective mechanisms for this problem.

## Troubleshooting

### File Not Found Error
- Ensure problem file exists at specified path
- Check file name spelling
- Verify relative path from script location

### Poor Results
- Increase `max_evaluations` for better convergence
- Increase `population_size` for more diversity
- Try different `dqn_learning_rate` values (0.0001 - 0.01)

### Slow Execution
- Reduce `population_size`
- Reduce `max_evaluations`
- Profile code to identify bottlenecks

## Further Reading

- `docs/MODQNHH_GUIDE.md`: Comprehensive algorithm guide
- `docs/MODQNHH_IMPLEMENTATION_SUMMARY.md`: Implementation details
- `evolu/optimizers/multiobjective/MODQNHH.py`: Source code
- `evolu/util/dqn.py`: DQN implementation

## Contact

For questions or issues:
- Check documentation in `docs/` directory
- Review source code comments
- Consult EvoSuite framework documentation
