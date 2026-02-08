# Quick Start Guide

## Installation

```bash
# Clone or navigate to the project directory
cd dac

# Install dependencies
pip install -r requirements.txt
```

## Running Examples

### Module 1: Molecular Physics

Calculate Boltzmann probabilities and analyze QM data:

```bash
python examples/example_molecular_physics.py
```

**What it does:**
- Calculates state probabilities from binding energies
- Analyzes temperature dependence
- Generates energy histograms
- Computes radial distribution functions

### Module 3: Markov Chain Degradation

Simulate long-term sorbent degradation:

```bash
python examples/example_markov_simulation.py
```

**What it does:**
- Simulates 1,000 to 10,000 cycles
- Tracks Active/Blocked/Lost states
- Validates against LCOC targets
- Visualizes degradation dynamics

### Module 4: Geospatial Performance

Analyze climate-specific performance:

```bash
python examples/example_geospatial_analysis.py
```

**What it does:**
- Generates synthetic MERRA-2 weather data
- Calculates location-specific CRR
- Optimizes operating conditions
- Compares multiple locations

### Integrated Analysis

Run complete multi-scale workflow:

```bash
python examples/integrated_example.py
```

**What it does:**
- Integrates all modules
- Analyzes molecular physics → degradation → geospatial
- Compares optimized vs baseline sorbents
- Generates comprehensive visualizations

## Basic Usage

### Molecular Physics

```python
from dac_framework import MolecularPhysics

mp = MolecularPhysics(temperature=298.15)
distribution = mp.calculate_motif_distribution()
print(distribution)
```

### Markov Degradation

```python
from dac_framework import MarkovDegradation

markov = MarkovDegradation()
results = markov.simulate_cycles(n_cycles=10000, record_interval=100)
validation = markov.validate_lcoc_target(results)
print(f"Decay constant: {validation['decay_constant']:.2e}")
```

### Geospatial Performance

```python
from dac_framework import (
    GeospatialPerformance,
    MERRA2DataLoader,
    SorbentParameters,
    Location
)

# Generate weather data
loader = MERRA2DataLoader()
weather = loader.generate_synthetic_data(n_hours=8760, location="temperate")

# Define sorbent
sorbent = SorbentParameters(
    co2_capacity_ref=2.5,
    working_capacity=1.8,
    h2o_penalty_factor=0.15,
    temp_coefficient=0.015,
    humidity_coefficient=1.2,
    regeneration_temp=100.0,
    regeneration_vacuum=50.0,
    cycle_time_base=6.0
)

# Analyze performance
geo = GeospatialPerformance()
location = Location("Site A", 33.45, -112.07)
geo.add_location(location, weather)
comparison = geo.compare_locations(sorbent)
print(comparison)
```

## Expected Outputs

Running the examples will generate:

- **Plots**: PNG files with visualizations
  - `energy_distribution.png` - Energy histogram
  - `rdf_analysis.png` - Radial distribution function
  - `mechanism_kinetics.png` - Arrhenius kinetics
  - `degradation_10k.png` - Degradation over cycles
  - `location_comparison.png` - Location performance
  - `hourly_performance.png` - Diurnal patterns
  - `integrated_analysis.png` - Complete workflow

- **Console Output**: Detailed analysis results and statistics

## Customization

### Custom Binding Motifs

```python
from dac_framework.molecular_physics import BindingMotif

custom_motif = BindingMotif(
    name="Custom Carbamate",
    delta_g=-28.0,  # kcal/mol
    is_reversible=True
)

mp.add_motif('custom', custom_motif)
```

### Custom Cycle Conditions

```python
from dac_framework.markov_degradation import CycleStage, CycleConditions

custom_cycle = [
    (CycleStage.ADSORPTION, CycleConditions(
        temperature=30, pressure=1013, humidity=0.7, duration=3600
    )),
    # ... more stages
]

results = markov.simulate_cycles(n_cycles=5000, cycle_stages=custom_cycle)
```

### Custom Weather Data

```python
import pandas as pd

# Load your own CSV with columns: timestamp, temperature, relative_humidity, pressure
weather_data = pd.read_csv('your_weather_data.csv')
geo.add_location(location, weather_data)
```

## Module 2: GNN Surrogate (Advanced)

Module 2 (GNN) requires training data and molecular structures. See the module documentation for details on:

- Preparing molecular datasets
- Training the GNN model
- Making predictions

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure you're running from the correct directory:

```bash
cd dac
python examples/example_molecular_physics.py
```

Or add the parent directory to your path:

```python
import sys
sys.path.append('..')
```

### PyTorch Geometric Installation

For Module 2 (GNN), if you have issues with PyTorch Geometric:

```bash
# Install PyTorch first
pip install torch

# Then install PyTorch Geometric
pip install torch-geometric
```

### RDKit Installation

```bash
# Via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit
```

## Next Steps

1. **Explore Examples**: Run all example scripts to understand each module
2. **Customize Parameters**: Modify sorbent parameters for your application
3. **Load Real Data**: Replace synthetic weather with MERRA-2 data
4. **Train GNN**: Prepare molecular datasets for Module 2
5. **Integrate**: Use the integrated_example.py as a template for your workflow

## Support

For questions or issues:
- Check the module documentation in each `.py` file
- Review the example scripts
- Examine the docstrings for detailed parameter descriptions

## References

- MERRA-2: [NASA GMAO](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
- PyTorch Geometric: [Documentation](https://pytorch-geometric.readthedocs.io/)
- RDKit: [Documentation](https://www.rdkit.org/docs/)
