# Beyond Isotherms: Humidity-Robust TVSA DAC Modeling

A multi-scale modeling framework for amine-based Direct Air Capture (DAC) systems that bridges molecular-scale physics with long-term cyclic performance modeling.

## Project Overview

This framework redefines performance and durability limits for amine-based DAC by integrating:
- Molecular dynamics and quantum mechanics (MD/QM)
- Machine learning-based sorbent design
- Long-term cyclic performance modeling (Markov Chains)
- Geospatial climate integration

## Architecture

### Module 1: Molecular Physics & Transition Logic
Convert quantum mechanical binding energies into probabilistic event statistics using Boltzmann distributions.

**Key Features:**
- 6-membered concerted mechanism for CO₂ capture
- Parsing of Gaussian/VASP outputs
- Energy distribution histograms
- State occupancy calculations

### Module 2: Data-Driven Design Surrogate
Equivariant Graph Neural Network (eGNN) for predicting sorbent performance across ambient climates.

**Key Features:**
- UMA architecture for 3D atomic positions
- Differential descriptor approach
- Predictions for CO₂ working capacity, H₂O co-adsorption penalty, and regeneration favorability

### Module 3: Cyclic Degradation Simulation
Discrete-time Markov Chain model simulating 1,000 to 10,000 capture cycles.

**Key Features:**
- Three-state system (Active, Blocked, Lost)
- Stage-specific transition probabilities
- Long-term degradation tracking
- LCOC target validation

### Module 4: Geospatial Performance & Siting
Integration with real-world hourly weather data (MERRA-2).

**Key Features:**
- Dynamic cycle duration adjustment
- Climate-specific performance optimization
- Carbon Removal Rate (CRR) calculation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from dac_framework import MolecularPhysics, GNNSurrogate, MarkovDegradation, GeospatialPerformance

# Module 1: Calculate state probabilities from QM data
mp = MolecularPhysics()
state_probs = mp.calculate_boltzmann_probabilities(energy_data)

# Module 2: Train GNN surrogate model
gnn = GNNSurrogate()
gnn.train(training_data)
predictions = gnn.predict(test_molecules)

# Module 3: Simulate degradation cycles
markov = MarkovDegradation()
results = markov.simulate_cycles(n_cycles=10000)

# Module 4: Calculate location-specific performance
geo = GeospatialPerformance()
crr = geo.calculate_crr(weather_data, sorbent_params)
```

## Project Structure

```
dac/
├── dac_framework/
│   ├── __init__.py
│   ├── molecular_physics.py      # Module 1
│   ├── gnn_surrogate.py          # Module 2
│   ├── markov_degradation.py     # Module 3
│   └── geospatial_performance.py # Module 4
├── examples/
│   ├── example_molecular_physics.py
│   ├── example_gnn_training.py
│   ├── example_markov_simulation.py
│   └── example_geospatial_analysis.py
├── data/
│   └── sample_data/
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- PyTorch, PyTorch Geometric
- RDKit, Mordred
- scikit-learn
- matplotlib, seaborn

## Citation

If you use this framework in your research, please cite:
```
[Project citation details]
```

## License

MIT License

## Contact

For questions and support, please open an issue on the GitHub repository.
