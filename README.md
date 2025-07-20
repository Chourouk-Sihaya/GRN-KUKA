# GRN-KUKA

# Gene Regulatory Network for Robotic Grasping
A bio-inspired approach for robot control (grasping) using Gene Regulatory Networks (GRNs). This system applies evolutionary optimization to evolve GRN-based controllers for KUKA robotic arm grasping tasks.
This project introduces a new paradigm for robot control by mimicking biological gene regulatory mechanisms. The system uses gene expression dynamics to generate motor commands, providing a natural hierarchical control structure with inherent temporal processing capabilities.

# Key Features
- Biologically-Inspired Control: Uses gene regulatory networks with sparse connectivity
- Evolutionary Optimization: Evolves GRN parameters using genetic algorithms
- Hierarchical Architecture: Natural gene expression cascades for complex behaviors
- Robustness: Built-in noise tolerance through biological mechanisms

# Algorithm
The GRN controller consists of:
1. Gene Expression Dynamics: Each gene has expression levels that change over time
2. Regulatory Interactions: Genes influence each other through weighted connections
3. Sensor Integration: Environmental inputs modulate gene expression
4. Motor Output: Gene expression levels are combined to generate robot actions

# Installation

```bash
pip install numpy torch torchvision matplotlib pillow
pip install pybullet pybullet_envs
pip install moviepy  # Optional, for video creation
```
For Google Colab
```
!pip install pybullet gymnasium pyvirtualdisplay moviepy opensimplex numpy matplotlib pillow scipy
```

# PyBullet KUKA Environment
The system requires the KUKA Diverse Object environment from PyBullet:

```python
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
```

# Usage
# Quick Start

```python
from grn_kuka_grasping import run_grn_experiment

# Run experiment with default parameters
results = run_grn_experiment(
    population_size=12,
    max_generations=15,
    test_episodes=30,
    create_video=True
)
```

# Custom Configuration
```python
# Create custom GRN genome
genome = GRNGenome(
    num_genes=50,
    num_actuators=7,
    sensor_dims=8
)

# Initialize optimizer
optimizer = GRNEvolutionaryOptimizer(
    population_size=16,
    elite_fraction=0.25,
    mutation_rate=0.2
)

# Run evolution
best_genome = optimizer.evolve(env, max_generations=20)
```

# Output Files
Each experiment generates:
- `grn_analysis.png`: Comprehensive analysis plots
- `best_grn_genome.pkl`: Evolved GRN parameters
- `test_results.pkl`: Detailed performance metrics
- `experiment_summary.json`: Summary statistics
- `grn_demo.mp4`: Demonstration video with gene expression overlay



## Contact

[Chourouk Guettas] - [g.chourouk@gmail.com]

Project Link: [https://github.com/Chourouk-Sihaya/GRN-KUKA](https://github.com/Chourouk-Sihaya/GRN-KUKA)
