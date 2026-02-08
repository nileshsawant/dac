"""
Configuration file for DAC Framework

This file contains default parameters and settings for all modules.
Modify these values to customize the framework for your application.
"""

# ============================================================================
# MODULE 1: MOLECULAR PHYSICS
# ============================================================================

MOLECULAR_PHYSICS = {
    # Default temperature for calculations (K)
    'default_temperature': 298.15,
    
    # Gas constant (kcal/(mol·K))
    'gas_constant': 8.314e-3,
    
    # Conversion factors
    'hartree_to_kcal': 627.509,
    'kcal_to_j': 4184,
    
    # Default binding motifs (kcal/mol)
    'binding_motifs': {
        'alkylammonium_carbamate': -26.5,
        'water_stabilized': -22.0,
        'irreversible_urea': -35.0,
    },
    
    # Activation energies (kcal/mol)
    'activation_energies': {
        'forward_capture': 15.0,
        'reverse_desorption': 41.5,
    }
}

# ============================================================================
# MODULE 2: GNN SURROGATE
# ============================================================================

GNN_SURROGATE = {
    # Model architecture
    'node_features': 64,
    'edge_features': 16,
    'hidden_dim': 128,
    'num_layers': 4,
    'num_targets': 3,  # CO2 capacity, H2O penalty, regen temp
    
    # Training parameters
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_epochs': 100,
    'train_val_split': 0.8,
    
    # Differential descriptors
    'use_differential': True,
    'descriptor_dim': 11,
}

# ============================================================================
# MODULE 3: MARKOV DEGRADATION
# ============================================================================

MARKOV_DEGRADATION = {
    # Initial state distribution [Active, Blocked, Lost]
    'initial_distribution': [1.0, 0.0, 0.0],
    
    # Chemical stressor parameters
    'stressors': {
        # C-N cleavage
        'ea_cn_cleavage': 20.0,  # kJ/mol
        
        # Water effects
        'water_block_rate': 1e-4,  # per cycle
        'water_recovery_rate': 0.1,  # per cycle
        
        # Oxidation
        'o2_concentration': 0.21,  # mole fraction
        'oxidation_rate_constant': 1e-6,
        
        # Urea formation
        'urea_formation_rate': 5e-5,
        'urea_reversal_rate': 1e-3,
        
        # Thermal degradation
        'thermal_degradation_rate': 1e-7,
        'thermal_ea': 100.0,  # kJ/mol
    },
    
    # S-TVSA cycle stages (default conditions)
    'cycle_stages': {
        'adsorption': {
            'temperature': 25,  # °C
            'pressure': 1013,  # mbar
            'humidity': 0.5,  # 0-1
            'duration': 3600,  # seconds
            'co2_partial_pressure': 0.4,  # mbar
        },
        'purging': {
            'temperature': 25,
            'pressure': 1013,
            'humidity': 0.2,
            'duration': 300,
            'co2_partial_pressure': 0.1,
        },
        'heating': {
            'temperature': 90,
            'pressure': 500,
            'humidity': 0.05,
            'duration': 1800,
            'co2_partial_pressure': 0.0,
        },
        'vacuum': {
            'temperature': 100,
            'pressure': 50,
            'humidity': 0.01,
            'duration': 1200,
            'co2_partial_pressure': 0.0,
        },
        'desorption': {
            'temperature': 100,
            'pressure': 50,
            'humidity': 0.01,
            'duration': 1800,
            'co2_partial_pressure': 0.0,
        },
        'cooling': {
            'temperature': 50,
            'pressure': 100,
            'humidity': 0.1,
            'duration': 900,
            'co2_partial_pressure': 0.0,
        },
        'pressurization': {
            'temperature': 25,
            'pressure': 1013,
            'humidity': 0.3,
            'duration': 300,
            'co2_partial_pressure': 0.2,
        },
    },
    
    # LCOC validation
    'lcoc_target': 100.0,  # $/tonne CO2
    'threshold_lambda': 5e-6,  # cycle^-1
}

# ============================================================================
# MODULE 4: GEOSPATIAL PERFORMANCE
# ============================================================================

GEOSPATIAL_PERFORMANCE = {
    # Default sorbent parameters
    'sorbent': {
        'co2_capacity_ref': 2.5,  # mol/kg at 25°C, 50% RH
        'working_capacity': 1.8,  # mol/kg
        'h2o_penalty_factor': 0.15,
        'temp_coefficient': 0.015,  # 1/°C
        'humidity_coefficient': 1.2,
        'regeneration_temp': 100.0,  # °C
        'regeneration_vacuum': 50.0,  # mbar
        'cycle_time_base': 6.0,  # hours
    },
    
    # Climate profiles for synthetic data generation
    'climates': {
        'temperate': {
            'temp_mean': 15,
            'temp_amp': 15,
            'rh_mean': 0.60,
        },
        'tropical': {
            'temp_mean': 27,
            'temp_amp': 5,
            'rh_mean': 0.80,
        },
        'arid': {
            'temp_mean': 25,
            'temp_amp': 20,
            'rh_mean': 0.25,
        },
        'polar': {
            'temp_mean': -5,
            'temp_amp': 10,
            'rh_mean': 0.70,
        },
    },
    
    # Optimization parameters
    'optimization': {
        # PSO parameters
        'pso_swarmsize': 30,
        'pso_maxiter': 100,
        
        # Parameter bounds [min, max]
        'cycle_time_bounds': [2.0, 12.0],  # hours
        'regen_temp_bounds': [80.0, 120.0],  # °C
        
        # Grid search resolution
        'grid_resolution': 7,
    },
    
    # Performance metrics
    'metrics': {
        # Energy model parameters (simplified)
        'heating_energy_per_degC': 2.5,  # kWh/tonne/°C
        'vacuum_energy_base': 50,  # kWh/tonne
        'water_removal_penalty': 100,  # kWh/tonne per unit RH
    },
}

# ============================================================================
# INTEGRATION SETTINGS
# ============================================================================

INTEGRATION = {
    # Number of cycles to simulate
    'n_cycles': 10000,
    
    # Recording interval
    'record_interval': 100,
    
    # Number of years for lifetime analysis
    'lifetime_years': 3,
    
    # Plotting parameters
    'plot_dpi': 300,
    'figure_format': 'png',
    
    # Parallel processing
    'use_multiprocessing': False,
    'n_workers': 4,
}

# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    'data_dir': 'data/',
    'sample_data_dir': 'data/sample_data/',
    'output_dir': 'output/',
    'figures_dir': 'output/figures/',
    'results_dir': 'output/results/',
}

# ============================================================================
# LOGGING
# ============================================================================

LOGGING = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'dac_framework.log',
}
