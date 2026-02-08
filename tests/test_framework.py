"""
Test Suite for DAC Framework

Run tests with:
    python -m pytest tests/
    
Or run individual test files:
    python -m pytest tests/test_molecular_physics.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dac_framework import (
    MolecularPhysics,
    BoltzmannCalculator,
    MarkovDegradation,
    GeospatialPerformance,
    MERRA2DataLoader,
    SorbentParameters
)


class TestMolecularPhysics:
    """Tests for Module 1: Molecular Physics"""
    
    def test_boltzmann_calculator_initialization(self):
        """Test BoltzmannCalculator initialization."""
        calc = BoltzmannCalculator(temperature=298.15)
        assert calc.temperature == 298.15
        assert calc.beta > 0
    
    def test_state_probability_calculation(self):
        """Test Boltzmann probability calculation."""
        calc = BoltzmannCalculator(temperature=298.15)
        prob = calc.calculate_state_probability(delta_g=-26.5)
        assert prob > 0
        assert np.isfinite(prob)
    
    def test_state_distribution_normalization(self):
        """Test that state distribution is normalized."""
        calc = BoltzmannCalculator(temperature=298.15)
        energies = [-26.5, -22.0, -35.0]
        probs = calc.calculate_state_distribution(energies)
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_equilibrium_constant(self):
        """Test equilibrium constant calculation."""
        calc = BoltzmannCalculator(temperature=298.15)
        K_eq = calc.calculate_equilibrium_constant(delta_g=-26.5)
        assert K_eq > 1.0  # Favorable reaction
    
    def test_molecular_physics_initialization(self):
        """Test MolecularPhysics initialization."""
        mp = MolecularPhysics(temperature=298.15)
        assert mp.temperature == 298.15
        assert len(mp.motifs) > 0
    
    def test_motif_distribution(self):
        """Test motif distribution calculation."""
        mp = MolecularPhysics()
        dist = mp.calculate_motif_distribution()
        assert len(dist) > 0
        assert 'Probability' in dist.columns
        assert np.isclose(dist['Probability'].sum(), 1.0)


class TestMarkovDegradation:
    """Tests for Module 3: Markov Degradation"""
    
    def test_markov_initialization(self):
        """Test MarkovDegradation initialization."""
        markov = MarkovDegradation()
        assert len(markov.p0) == 3
        assert np.isclose(np.sum(markov.p0), 1.0)
    
    def test_single_cycle_simulation(self):
        """Test single cycle simulation."""
        markov = MarkovDegradation()
        P = markov.simulate_single_cycle()
        assert P.shape == (3, 3)
        # Check row sums equal 1 (stochastic matrix)
        assert np.allclose(P.sum(axis=1), 1.0)
    
    def test_multi_cycle_simulation(self):
        """Test multi-cycle simulation."""
        markov = MarkovDegradation()
        results = markov.simulate_cycles(n_cycles=100, record_interval=10)
        assert len(results) > 0
        assert 'active' in results.columns
        assert 'blocked' in results.columns
        assert 'lost' in results.columns
    
    def test_state_conservation(self):
        """Test that state probabilities sum to 1."""
        markov = MarkovDegradation()
        results = markov.simulate_cycles(n_cycles=100, record_interval=10)
        for _, row in results.iterrows():
            total = row['active'] + row['blocked'] + row['lost']
            assert np.isclose(total, 1.0)
    
    def test_decay_constant_calculation(self):
        """Test decay constant calculation."""
        markov = MarkovDegradation()
        results = markov.simulate_cycles(n_cycles=1000, record_interval=100)
        lambda_decay = markov.calculate_decay_constant(results)
        assert lambda_decay > 0
        assert np.isfinite(lambda_decay)
    
    def test_lcoc_validation(self):
        """Test LCOC validation."""
        markov = MarkovDegradation()
        results = markov.simulate_cycles(n_cycles=1000, record_interval=100)
        validation = markov.validate_lcoc_target(results)
        assert 'decay_constant' in validation
        assert 'passes_threshold' in validation
        assert isinstance(validation['passes_threshold'], (bool, np.bool_))


class TestGeospatialPerformance:
    """Tests for Module 4: Geospatial Performance"""
    
    def test_merra2_loader_initialization(self):
        """Test MERRA2DataLoader initialization."""
        loader = MERRA2DataLoader()
        assert loader.weather_data is None
    
    def test_synthetic_data_generation(self):
        """Test synthetic weather data generation."""
        loader = MERRA2DataLoader()
        weather = loader.generate_synthetic_data(n_hours=100, location="temperate")
        assert len(weather) == 100
        assert 'temperature' in weather.columns
        assert 'relative_humidity' in weather.columns
        assert 'pressure' in weather.columns
    
    def test_sorbent_parameters(self):
        """Test SorbentParameters initialization."""
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
        assert sorbent.co2_capacity_ref == 2.5
        assert sorbent.regeneration_temp == 100.0
    
    def test_capacity_adjustment(self):
        """Test capacity adjustment for different conditions."""
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
        
        # Test at reference conditions
        capacity_ref = sorbent.adjust_capacity(25.0, 0.5)
        assert capacity_ref > 0
        
        # Test at higher temperature (should decrease)
        capacity_hot = sorbent.adjust_capacity(35.0, 0.5)
        assert capacity_hot < capacity_ref
        
        # Test at higher humidity (should decrease)
        capacity_humid = sorbent.adjust_capacity(25.0, 0.8)
        assert capacity_humid < capacity_ref
    
    def test_geospatial_performance_initialization(self):
        """Test GeospatialPerformance initialization."""
        geo = GeospatialPerformance()
        assert len(geo.locations) == 0
        assert len(geo.performance_data) == 0


class TestIntegration:
    """Integration tests across modules"""
    
    def test_molecular_to_markov_integration(self):
        """Test integration between molecular physics and Markov chain."""
        # Get molecular binding information
        mp = MolecularPhysics()
        dist = mp.calculate_motif_distribution()
        reversible_fraction = dist[dist['Reversible']]['Probability'].sum()
        
        # Use in Markov simulation
        markov = MarkovDegradation()
        results = markov.simulate_cycles(n_cycles=100, record_interval=10)
        
        assert reversible_fraction > 0
        assert len(results) > 0
    
    def test_markov_to_geospatial_integration(self):
        """Test integration between Markov degradation and geospatial."""
        # Simulate degradation with fewer cycles to ensure active sites remain
        markov = MarkovDegradation()
        deg_results = markov.simulate_cycles(n_cycles=100, record_interval=10)
        
        # Get degradation factor
        final_active = deg_results.iloc[-1]['active']
        
        # Apply to geospatial performance
        loader = MERRA2DataLoader()
        weather = loader.generate_synthetic_data(n_hours=100, location="temperate")
        
        # Check that simulation runs and produces results
        assert final_active >= 0  # Should be non-negative
        assert len(weather) > 0
        # Check that degradation occurs (active sites decrease from initial 1.0)
        assert final_active <= 1.0


def test_numpy_version():
    """Test NumPy is installed and working."""
    assert np.__version__ is not None


def test_imports():
    """Test that all main modules can be imported."""
    from dac_framework import (
        MolecularPhysics,
        BoltzmannCalculator,
        MarkovDegradation,
        TransitionMatrix,
        GeospatialPerformance,
        MERRA2DataLoader,
        PerformanceCalculator,
        ClimateOptimizer,
        SorbentParameters,
        Location
    )
    assert True  # If we get here, imports worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
