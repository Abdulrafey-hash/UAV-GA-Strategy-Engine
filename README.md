# Multi-Objective UAV Fleet Strategy Engine using Genetic Algorithm (GA)

**Course:** Artificial Intelligence (BSCS-5) | **Complex Computing Problem (CCP) - Assignment 2**  
**Institution:** Department of Computer Science, Bahria University, Lahore Campus  

## 1. Project Overview
This repository implements an object-oriented **Genetic Algorithm (GA) Strategy Engine** for autonomous Unmanned Aerial Vehicle (UAV) fleet surveillance and disaster management over a discretized $50 \times 50$ hazard grid. The system optimizes multi-drone deployment trajectories, speed profiles, and active sensor schedules across competing operational objectives:
- **Maximization:** High-priority disaster zone coverage and target detection rate.
- **Minimization:** Total battery/energy expenditure, mission flight completion time, and cumulative exposure to simulated environmental hazard pockets (radiation/high-wind zones).
- **Hard Constraint Handling:** Dynamic geometric waypoint repair and penalty mechanisms for restricted No-Fly Zones (NFZ) and maximum battery capacity limits.

## 2. Repository Structure
```text
UAV-GA-Strategy-Engine/
├── environment.py       # 50x50 Grid Environment, Priority Clusters, Hazard Pockets & No-Fly Zones
├── chromosome.py        # Hybrid Multi-Segment Chromosome & Dynamic Geometric Repair Mechanism
├── ga_engine.py         # Core Genetic Algorithm (Tournament Selection, BLX-alpha Crossover, Mutation)
├── main.py              # Simulation Entry Point, Baseline Comparison & Plot Generation
├── requirements.txt     # Python Dependencies
└── README.md            # Documentation and Execution Guide
```

## 3. Installation & Execution Instructions
1. Clone this repository and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the complete simulation and generate output visualizations:
   ```bash
   python main.py
   ```
3. Upon completion, the script outputs generation-by-generation fitness logs in the terminal and saves high-resolution visual plots (`convergence_curve.png`, `uav_trajectories_heatmap.png`, and `sensitivity_analysis.png`) in the working directory.
