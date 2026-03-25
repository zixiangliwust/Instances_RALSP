# Learning-based Multi-Objective Hyper-Heuristic Algorithm for Reconfigurable Assembly Line Scheduling Problems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository contains benchmark instances and supporting materials for the research paper **"Learning-based multi-objective hyper-heuristic algorithm for reconfigurable assembly line scheduling problems"**. 

The study addresses the **Reconfigurable Assembly Line Scheduling Problem (RALSP)**, which involves optimizing product sequences to minimize:
- **Reconfiguration cost**
- **Production workload equalization** 
- **Logistics leveling**

## 🎯 Research Contributions

### 1. Novel Mathematical Model
- A fully linearized multi-objective mixed integer linear programming (MILP) model
- Rectifies deficiencies in prior formulations with nonlinearities and logical inconsistencies
- Directly solvable by standard optimization tools

### 2. Q-Learning Based Multi-Objective Hyper-Heuristic (QMOHH)
The proposed algorithm integrates multiple metaheuristic operators within a unified search framework:
- **Particle Swarm Optimization (PSO)**
- **Teaching-Learning-Based Optimization (TLBO)**
- **Whale Optimization Algorithm (WOA)**
- **Grey Wolf Optimizer (GWO)**

Key features:
- **Q-learning** for adaptive operator selection based on real-time performance feedback
- **Density-aware leader selection strategy** with survival time decay factor
- Balances exploration and exploitation effectively

### 3. Comprehensive Evaluation
- Outperforms 9 state-of-the-art multi-objective algorithms
- Tested on 100 generated benchmark instances
- Validated using three performance indicators

## 📁 Repository Structure

```
Instances_RALSP/
├── Instances/              # Benchmark problem instances
│   ├── example_of_english_paper.txt    # Illustrative example from the paper
│   ├── example_of_paper.txt            # Additional example
│   └── experimental_data_100_*.txt     # 100 benchmark instances (1-100)
├── .qoder/                 # Project configuration
└── LICENSE                 # MIT License
```

## 📊 Instance Format

Each instance file follows a standardized format:

```
<number of products>
[product count]

<product cycle>
[product_id] [cycle_time]

<number of workstations>
[workstation count]

<station length>
[length_1] [length_2] ... [length_n]

<working hours>
[product_id] [hour_1] [hour_2] ... [hour_m]

<product switching cost>
[from,to] [cost]

<number of parts>
[part count]

<optional frequency>
[part_id] [frequency]

<product assembly parts>
[product_id] [part_1_flag] [part_2_flag] ...
```

## 🔬 Problem Description

### Reconfigurable Assembly Line Configuration

Unlike traditional assembly lines, reconfigurable assembly lines can:
- Produce multi-variety and small-lot products
- Swiftly switch between different products
- Combine high throughput rates with high flexibility

### Three Assembly Line Types

1. **Single-model assembly line**: Produces one product type only
2. **Mixed-model assembly line**: Assembles similar products simultaneously (negligible reconfiguration cost)
3. **Multi-model assembly line**: Produces different products in batches (requires physical reconfiguration)

### Optimization Objectives

The RALSP aims to optimize three conflicting objectives simultaneously:

1. **Minimize Reconfiguration Cost**: Reduce costs associated with switching between product batches
2. **Equalize Production Workload**: Balance workload across workstations
3. **Level Logistics**: Smooth material flow and parts consumption

## 🧪 Computational Results

### Comparison Algorithms

The QMOHH was compared against nine high-performing algorithms:
- GDE3 (Generalized Differential Evolution)
- MOEAD (Multi-Objective Evolutionary Algorithm based on Decomposition)
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- And six others

### Key Findings

- The ε-constraint method successfully generates Pareto optimal solutions
- QMOHH consistently outperforms competing algorithms across all performance metrics
- The integration of Q-learning significantly improves search efficiency
- Density-aware leader selection enhances solution diversity

## 👥 Authors & Affiliations

**Haoyi Zhao**¹,²,³, **Xiangming Huang**¹, **Guoliang Liu**³, **Zixiang Li**⁴*, **Fan Chen**², **Gaojie Lu**²

1. College of Mechanical and Vehicle Engineering, Hunan University, Changsha, Hunan, China
2. Wuhan Vocational College of Software and Engineering (Wuhan Open University), Wuhan, Hubei, China
3. Hunan Sinoboom Intelligent Equipment Co., Ltd., Changsha, Hunan, China
4. Key Laboratory of Metallurgical Equipment and Control Technology of Ministry of Education, Wuhan University of Science and Technology, Wuhan, Hubei, China

**Corresponding Author**: Zixiang Li (lizixiang@wust.edu.cn)

### Contact Information

- Haoyi Zhao: zhaoya@whvcse.edu.cn
- Xiangming Huang: h_xiangming@aliyun.com
- Guoliang Liu: steven.liu@sinoboom.com.cn
- Zixiang Li: lizixiang@wust.edu.cn
- Fan Chen: 42100391@whvcse.edu.cn
- Gaojie Lu: 635791287@qq.com

## 📝 Citation

**Status**: Paper submitted (not yet accepted)

When referencing this work, please cite as:

```bibtex
@article{zhao2026qmoxx,
  title={Learning-based multi-objective hyper-heuristic algorithm for reconfigurable assembly line scheduling problems},
  author={Zhao, Haoyi and Huang, Xiangming and Liu, Guoliang and Li, Zixiang and Chen, Fan and Lu, Gaojie},
  journal={Under Review},
  year={2026}
}
```

## 🔑 Keywords

- Reconfigurable assembly line
- Hyper-heuristic algorithm
- Multi-objective optimization
- Scheduling
- Q-learning
- Metaheuristic algorithm

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Zixiang Li

## 🙏 Acknowledgments

The authors would like to thank the reviewers and editors for their valuable comments and suggestions that improved the quality of this paper.

## 📚 Paper Organization

1. **Introduction** - Background and research gaps
2. **Literature Review** - Related work on RALSP, hyper-heuristics, and Q-learning
3. **Problem Description** - Mathematical formulation
4. **Proposed QMOHH** - Algorithm details
5. **Numerical Case Study** - Illustrative example
6. **Computational Study** - Performance evaluation
7. **Conclusion** - Summary and future research directions

## 💻 Usage

The benchmark instances in the `Instances/` directory can be used to:
- Test and validate optimization algorithms for RALSP
- Compare performance of multi-objective algorithms
- Reproduce experimental results from the paper
- Develop new solution approaches for reconfigurable manufacturing systems

## 🔮 Future Research Directions

The paper suggests several promising avenues for future work:
- Extension to dynamic environments with uncertain demand
- Integration with sustainability objectives (energy consumption, carbon footprint)
- Application to larger-scale industrial cases
- Exploration of deep reinforcement learning techniques
- Consideration of worker skills and ergonomics

---

**Note**: This repository will be updated upon paper acceptance with additional materials including source code, detailed experimental results, and implementation guidelines.
