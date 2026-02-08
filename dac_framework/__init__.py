"""
Beyond Isotherms: Multi-scale DAC Modeling Framework
=====================================================

A comprehensive framework for modeling humidity-robust Temperature-Vacuum Swing 
Adsorption (TVSA) Direct Air Capture systems.

Modules:
--------
- molecular_physics: Quantum mechanical binding energies and transition logic
- gnn_surrogate: Graph Neural Network for sorbent performance prediction
- markov_degradation: Cyclic degradation simulation using Markov Chains
- geospatial_performance: Climate-integrated performance modeling
"""

__version__ = "0.1.0"
__author__ = "DAC Research Team"

from .molecular_physics import MolecularPhysics, BoltzmannCalculator, BindingMotif
from .gnn_surrogate import GNNSurrogate, DifferentialDescriptor, EGNNLayer
from .markov_degradation import MarkovDegradation, TransitionMatrix, StressorParameters
from .geospatial_performance import GeospatialPerformance, ClimateOptimizer, MERRA2DataLoader, PerformanceCalculator, SorbentParameters, Location

__all__ = [
    "MolecularPhysics",
    "BoltzmannCalculator",
    "BindingMotif",
    "GNNSurrogate",
    "DifferentialDescriptor",
    "EGNNLayer",
    "MarkovDegradation",
    "TransitionMatrix",
    "StressorParameters",
    "SorbentParameters",
    "GeospatialPerformance",
    "ClimateOptimizer",
    "MERRA2DataLoader",
    "PerformanceCalculator",
    "Location",
]
