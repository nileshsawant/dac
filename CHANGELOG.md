# Changelog

All notable changes to the Beyond Isotherms DAC Framework will be documented in this file.

## [0.1.0] - 2026-02-08

### Added
- **Module 1: Molecular Physics & Transition Logic**
  - Boltzmann distribution calculator for state probabilities
  - Gaussian and VASP output file parsers
  - Energy distribution histogram generation
  - Radial Distribution Function (RDF) analysis
  - 6-membered concerted mechanism kinetics
  - Temperature-dependent equilibrium calculations
  - Binding motif management system

- **Module 2: GNN Design Surrogate**
  - Equivariant Graph Neural Network (EGNN) implementation
  - Differential descriptor calculator (reactant-product)
  - Multi-target prediction framework
  - RDKit molecular descriptor integration
  - PyTorch Geometric data conversion utilities
  - Training and inference pipeline

- **Module 3: Markov Chain Degradation Simulation**
  - Three-state Markov model (Active/Blocked/Lost)
  - Seven S-TVSA cycle stages with customizable conditions
  - Chemical stressor-based transition probability calculator
  - 10,000+ cycle simulation capability
  - Exponential decay analysis
  - LCOC validation against $100/tonne CO₂ target
  - Stage-by-stage transition matrix analysis

- **Module 4: Geospatial Performance & Siting**
  - MERRA-2 weather data loader (CSV and netCDF)
  - Synthetic weather data generator for 4 climate zones
  - Carbon Removal Rate (CRR) calculator
  - Dynamic cycle time adjustment
  - Particle Swarm Optimization (PSO) for operating conditions
  - Grid search optimizer
  - Multi-location performance comparison
  - Hourly and seasonal analysis tools

- **Supporting Infrastructure**
  - Centralized configuration system (`config.py`)
  - Utility functions for statistics, plotting, and analysis
  - Comprehensive test suite with 30+ test cases
  - Data validation utilities
  - LCOC calculation tools
  - Summary report generation

- **Documentation**
  - README.md with project overview
  - QUICKSTART.md for quick installation and usage
  - PROJECT_SUMMARY.md with detailed technical information
  - Inline documentation for all classes and functions
  - Example scripts for each module
  - Integrated workflow demonstration

- **Examples**
  - `example_molecular_physics.py`: 6 comprehensive examples
  - `example_markov_simulation.py`: 6 simulation scenarios
  - `example_geospatial_analysis.py`: 6 climate analysis examples
  - `integrated_example.py`: Complete multi-scale workflow

### Features
- Publication-ready visualizations (300 DPI)
- Multi-scale integration (molecular → geospatial)
- Climate-aware performance modeling
- Long-term degradation tracking
- Energy-efficiency trade-off analysis
- Location-specific siting recommendations

### Technical Specifications
- Python 3.8+ compatibility
- NumPy/SciPy-based calculations
- PyTorch/PyTorch Geometric for ML
- RDKit for molecular descriptors
- Matplotlib/Seaborn for visualization
- Pandas for data management

### Validation
- Boltzmann distribution normalization tests
- Markov chain probability conservation tests
- Capacity adjustment validation
- Weather data format validation
- Integration tests across modules

## [Unreleased]

### Planned Features
- Pre-trained GNN models with example datasets
- Direct MERRA-2 API integration
- Detailed economic analysis module
- Uncertainty quantification tools
- Web-based interactive dashboard
- Multi-contaminant effects modeling

### Known Limitations
- Module 2 (GNN) requires custom training data
- Synthetic weather data only (no real MERRA-2 files included)
- Simplified energy model
- Single-component CO₂ capture (no NOₓ/SOₓ)

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
