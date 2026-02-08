"""
Module 3: Cyclic Degradation Simulation (Markov Chain)
======================================================

This module implements a discrete-time Markov Chain model to simulate
1,000 to 10,000 capture cycles and track sorbent degradation.

Key Features:
- Three-state system: Active, Blocked, Lost
- Stage-specific transition probabilities
- S-TVSA cycle simulation (Adsorption, Purging, Heating, Vacuum, Desorption, Cooling, Pressurization)
- Long-term performance tracking
- LCOC validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import seaborn as sns


class SiteState(Enum):
    """Enumeration of sorbent site states."""
    ACTIVE = 0      # Healthy sites capable of reversible adsorption
    BLOCKED = 1     # Temporarily inaccessible (water clustering, reversible urea)
    LOST = 2        # Permanently degraded (oxidative C-N cleavage, pore collapse)


class CycleStage(Enum):
    """Enumeration of S-TVSA cycle stages."""
    ADSORPTION = "adsorption"
    PURGING = "purging"
    HEATING = "heating"
    VACUUM = "vacuum"
    DESORPTION = "desorption"
    COOLING = "cooling"
    PRESSURIZATION = "pressurization"


@dataclass
class CycleConditions:
    """Operating conditions for a cycle stage."""
    temperature: float  # Temperature in °C
    pressure: float  # Pressure in mbar
    humidity: float  # Relative humidity (0-1)
    duration: float  # Duration in seconds
    co2_partial_pressure: float = 0.4  # CO2 partial pressure in mbar (400 ppm)


@dataclass
class StressorParameters:
    """Chemical stressor parameters affecting transition probabilities."""
    # C-N cleavage kinetics
    ea_cn_cleavage: float = 20.0  # Activation energy (kJ/mol)
    
    # Water effects
    water_block_rate: float = 1e-4  # Rate of water blocking per cycle
    water_recovery_rate: float = 0.1  # Rate of water desorption
    
    # Oxidation
    o2_concentration: float = 0.21  # O2 mole fraction
    oxidation_rate_constant: float = 1e-6  # Base oxidation rate
    
    # Urea formation
    urea_formation_rate: float = 5e-5  # Rate of urea formation
    urea_reversal_rate: float = 1e-3  # Rate of urea reversal
    
    # Thermal degradation
    thermal_degradation_rate: float = 1e-7  # Base thermal degradation rate
    thermal_ea: float = 100.0  # Activation energy for thermal degradation (kJ/mol)


class TransitionMatrix:
    """
    Calculate and manage transition probability matrices for Markov Chain.
    """
    
    def __init__(self, stressors: Optional[StressorParameters] = None):
        """
        Initialize transition matrix calculator.
        
        Parameters:
        -----------
        stressors : StressorParameters, optional
            Chemical stressor parameters
        """
        self.stressors = stressors or StressorParameters()
        self.R = 8.314  # Gas constant (J/(mol·K))
        
    def arrhenius_rate(self, ea: float, temperature: float, 
                      pre_exponential: float = 1.0) -> float:
        """
        Calculate rate constant using Arrhenius equation.
        
        k = A * exp(-Ea/RT)
        
        Parameters:
        -----------
        ea : float
            Activation energy (kJ/mol)
        temperature : float
            Temperature (°C)
        pre_exponential : float
            Pre-exponential factor
            
        Returns:
        --------
        float
            Rate constant
        """
        T_kelvin = temperature + 273.15
        ea_joules = ea * 1000  # Convert kJ/mol to J/mol
        return pre_exponential * np.exp(-ea_joules / (self.R * T_kelvin))
    
    def calculate_cn_cleavage_probability(self, 
                                         conditions: CycleConditions) -> float:
        """
        Calculate probability of C-N cleavage based on conditions.
        
        Enhanced by CO2 presence and temperature.
        """
        base_rate = self.arrhenius_rate(
            self.stressors.ea_cn_cleavage,
            conditions.temperature
        )
        
        # CO2 catalytic effect
        co2_factor = 1.0 + 10.0 * (conditions.co2_partial_pressure / 0.4)
        
        # Convert rate to probability (assuming exponential process)
        probability = 1 - np.exp(-base_rate * co2_factor * conditions.duration)
        
        return probability
    
    def calculate_water_blocking_probability(self,
                                            conditions: CycleConditions) -> float:
        """
        Calculate probability of water blocking active sites.
        
        Depends on humidity and temperature.
        """
        if conditions.humidity < 0.01:
            return 0.0
        
        # Water adsorption increases with humidity
        humidity_factor = conditions.humidity ** 2
        
        # Temperature reduces blocking (enhances desorption)
        temp_factor = np.exp(-(conditions.temperature - 25) / 50)
        
        probability = (self.stressors.water_block_rate * 
                      humidity_factor * temp_factor * conditions.duration)
        
        return min(probability, 1.0)
    
    def calculate_water_recovery_probability(self,
                                            conditions: CycleConditions) -> float:
        """
        Calculate probability of recovering blocked sites by water desorption.
        
        Enhanced by temperature and vacuum.
        """
        # Temperature enhancement
        temp_factor = np.exp((conditions.temperature - 25) / 50)
        
        # Vacuum enhancement (low pressure removes water)
        vacuum_factor = np.exp(-conditions.pressure / 100)
        
        probability = (self.stressors.water_recovery_rate * 
                      temp_factor * vacuum_factor * conditions.duration)
        
        return min(probability, 1.0)
    
    def calculate_oxidation_probability(self,
                                       conditions: CycleConditions) -> float:
        """
        Calculate probability of oxidative degradation.
        
        Enhanced by oxygen presence and temperature.
        """
        # Oxygen availability factor
        o2_factor = self.stressors.o2_concentration
        
        # Temperature enhancement
        temp_factor = self.arrhenius_rate(50.0, conditions.temperature)
        
        probability = (self.stressors.oxidation_rate_constant * 
                      o2_factor * temp_factor * conditions.duration)
        
        return min(probability, 1.0)
    
    def calculate_urea_formation_probability(self,
                                            conditions: CycleConditions) -> float:
        """
        Calculate probability of urea formation (reversible blocking).
        
        Occurs during CO2 capture at moderate temperatures.
        """
        if conditions.temperature > 80:
            return 0.0  # Urea unstable at high temperature
        
        co2_factor = conditions.co2_partial_pressure / 0.4
        
        probability = (self.stressors.urea_formation_rate * 
                      co2_factor * conditions.duration)
        
        return min(probability, 1.0)
    
    def calculate_urea_reversal_probability(self,
                                           conditions: CycleConditions) -> float:
        """
        Calculate probability of urea reversal (unblocking).
        
        Enhanced by high temperature during regeneration.
        """
        if conditions.temperature < 60:
            return 0.0
        
        temp_factor = np.exp((conditions.temperature - 80) / 30)
        
        probability = (self.stressors.urea_reversal_rate * 
                      temp_factor * conditions.duration)
        
        return min(probability, 1.0)
    
    def calculate_thermal_degradation_probability(self,
                                                  conditions: CycleConditions) -> float:
        """
        Calculate probability of thermal degradation.
        
        Occurs at high regeneration temperatures.
        """
        if conditions.temperature < 80:
            return 0.0
        
        rate = self.arrhenius_rate(
            self.stressors.thermal_ea,
            conditions.temperature,
            self.stressors.thermal_degradation_rate
        )
        
        probability = 1 - np.exp(-rate * conditions.duration)
        
        return min(probability, 1.0)
    
    def build_transition_matrix(self, 
                               stage: CycleStage,
                               conditions: CycleConditions) -> np.ndarray:
        """
        Build 3x3 transition probability matrix for a cycle stage.
        
        Matrix structure: P[i,j] = probability of transitioning from state i to state j
        States: 0 = Active, 1 = Blocked, 2 = Lost
        
        Parameters:
        -----------
        stage : CycleStage
            Current cycle stage
        conditions : CycleConditions
            Operating conditions
            
        Returns:
        --------
        np.ndarray
            3x3 transition probability matrix
        """
        P = np.zeros((3, 3))
        
        # Calculate relevant probabilities
        p_cn = self.calculate_cn_cleavage_probability(conditions)
        p_oxidation = self.calculate_oxidation_probability(conditions)
        p_thermal = self.calculate_thermal_degradation_probability(conditions)
        p_water_block = self.calculate_water_blocking_probability(conditions)
        p_water_recovery = self.calculate_water_recovery_probability(conditions)
        p_urea_form = self.calculate_urea_formation_probability(conditions)
        p_urea_rev = self.calculate_urea_reversal_probability(conditions)
        
        # Total degradation probability (Active -> Lost)
        p_degrade = min(p_cn + p_oxidation + p_thermal, 1.0)
        
        # Active -> X transitions
        if stage in [CycleStage.ADSORPTION, CycleStage.PRESSURIZATION]:
            # During adsorption: can block or degrade
            P[0, 1] = min(p_water_block + p_urea_form, 1.0 - p_degrade)
            P[0, 2] = p_degrade
            P[0, 0] = 1.0 - P[0, 1] - P[0, 2]
            
        elif stage in [CycleStage.HEATING, CycleStage.DESORPTION, CycleStage.VACUUM]:
            # During regeneration: can recover blocked sites or degrade
            P[0, 2] = p_degrade
            P[0, 0] = 1.0 - P[0, 2]
            
        else:
            # During purging/cooling: minimal changes
            P[0, 2] = p_degrade * 0.5
            P[0, 0] = 1.0 - P[0, 2]
        
        # Blocked -> X transitions
        if stage in [CycleStage.HEATING, CycleStage.DESORPTION, CycleStage.VACUUM]:
            # Regeneration can recover blocked sites
            P[1, 0] = min(p_water_recovery + p_urea_rev, 1.0 - p_degrade)
            P[1, 2] = p_degrade
            P[1, 1] = 1.0 - P[1, 0] - P[1, 2]
        else:
            # Otherwise, blocked sites remain blocked or degrade
            P[1, 2] = p_degrade * 0.5  # Slightly less prone to degradation
            P[1, 1] = 1.0 - P[1, 2]
        
        # Lost -> X transitions (absorbing state)
        P[2, 2] = 1.0
        
        return P


class MarkovDegradation:
    """
    Discrete-time Markov Chain model for cyclic degradation simulation.
    """
    
    def __init__(self, 
                 initial_distribution: Optional[np.ndarray] = None,
                 stressors: Optional[StressorParameters] = None):
        """
        Initialize Markov degradation model.
        
        Parameters:
        -----------
        initial_distribution : np.ndarray, optional
            Initial state distribution [p_active, p_blocked, p_lost]
            Default: [1.0, 0.0, 0.0] (all sites active)
        stressors : StressorParameters, optional
            Chemical stressor parameters
        """
        if initial_distribution is None:
            self.p0 = np.array([1.0, 0.0, 0.0])
        else:
            self.p0 = np.array(initial_distribution)
        
        self.transition_calc = TransitionMatrix(stressors)
        self.history = []
        
    def define_cycle_stages(self) -> List[Tuple[CycleStage, CycleConditions]]:
        """
        Define standard S-TVSA cycle stages and conditions.
        
        Returns:
        --------
        List[Tuple[CycleStage, CycleConditions]]
            List of (stage, conditions) tuples
        """
        stages = [
            (CycleStage.ADSORPTION, CycleConditions(
                temperature=25, pressure=1013, humidity=0.5, duration=3600,
                co2_partial_pressure=0.4
            )),
            (CycleStage.PURGING, CycleConditions(
                temperature=25, pressure=1013, humidity=0.2, duration=300,
                co2_partial_pressure=0.1
            )),
            (CycleStage.HEATING, CycleConditions(
                temperature=90, pressure=500, humidity=0.05, duration=1800,
                co2_partial_pressure=0.0
            )),
            (CycleStage.VACUUM, CycleConditions(
                temperature=100, pressure=50, humidity=0.01, duration=1200,
                co2_partial_pressure=0.0
            )),
            (CycleStage.DESORPTION, CycleConditions(
                temperature=100, pressure=50, humidity=0.01, duration=1800,
                co2_partial_pressure=0.0
            )),
            (CycleStage.COOLING, CycleConditions(
                temperature=50, pressure=100, humidity=0.1, duration=900,
                co2_partial_pressure=0.0
            )),
            (CycleStage.PRESSURIZATION, CycleConditions(
                temperature=25, pressure=1013, humidity=0.3, duration=300,
                co2_partial_pressure=0.2
            )),
        ]
        
        return stages
    
    def simulate_single_cycle(self, 
                             cycle_stages: Optional[List[Tuple[CycleStage, CycleConditions]]] = None
                             ) -> np.ndarray:
        """
        Simulate a single complete cycle.
        
        Parameters:
        -----------
        cycle_stages : List[Tuple[CycleStage, CycleConditions]], optional
            Custom cycle stages (default: standard S-TVSA)
            
        Returns:
        --------
        np.ndarray
            Transition matrix for complete cycle
        """
        if cycle_stages is None:
            cycle_stages = self.define_cycle_stages()
        
        # Compute composite transition matrix
        P_cycle = np.eye(3)
        
        for stage, conditions in cycle_stages:
            P_stage = self.transition_calc.build_transition_matrix(stage, conditions)
            P_cycle = P_cycle @ P_stage
        
        return P_cycle
    
    def simulate_cycles(self,
                       n_cycles: int,
                       cycle_stages: Optional[List[Tuple[CycleStage, CycleConditions]]] = None,
                       record_interval: int = 1) -> pd.DataFrame:
        """
        Simulate multiple cycles and track state evolution.
        
        p_n = p_0 @ P^n
        
        Parameters:
        -----------
        n_cycles : int
            Number of cycles to simulate
        cycle_stages : List[Tuple[CycleStage, CycleConditions]], optional
            Custom cycle stages
        record_interval : int
            Record state every N cycles (default: 1)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with cycle number and state distributions
        """
        P_cycle = self.simulate_single_cycle(cycle_stages)
        
        # Initialize
        p_current = self.p0.copy()
        self.history = [{'cycle': 0, 'active': p_current[0], 
                        'blocked': p_current[1], 'lost': p_current[2]}]
        
        # Simulate
        for cycle in range(1, n_cycles + 1):
            p_current = p_current @ P_cycle
            
            if cycle % record_interval == 0:
                self.history.append({
                    'cycle': cycle,
                    'active': p_current[0],
                    'blocked': p_current[1],
                    'lost': p_current[2]
                })
        
        return pd.DataFrame(self.history)
    
    def calculate_decay_constant(self, results: pd.DataFrame) -> float:
        """
        Calculate exponential decay constant for active sites.
        
        Active(n) ≈ exp(-λn)
        
        Parameters:
        -----------
        results : pd.DataFrame
            Simulation results
            
        Returns:
        --------
        float
            Decay constant λ (cycle^-1)
        """
        cycles = results['cycle'].values
        active = results['active'].values
        
        # Fit exponential decay
        # ln(Active) = ln(Active_0) - λ*n
        log_active = np.log(active + 1e-10)  # Avoid log(0)
        
        # Linear regression
        coeffs = np.polyfit(cycles, log_active, 1)
        lambda_decay = -coeffs[0]
        
        return lambda_decay
    
    def validate_lcoc_target(self, 
                            results: pd.DataFrame,
                            target_lcoc: float = 100.0,
                            threshold_lambda: float = 5e-6) -> Dict:
        """
        Validate against LCOC (Levelized Cost of Carbon) target.
        
        Parameters:
        -----------
        results : pd.DataFrame
            Simulation results
        target_lcoc : float
            Target LCOC ($/tonne CO2)
        threshold_lambda : float
            Maximum acceptable decay constant (cycle^-1)
            
        Returns:
        --------
        Dict
            Validation results
        """
        lambda_decay = self.calculate_decay_constant(results)
        
        # Calculate active fraction after 10,000 cycles
        active_10k = results[results['cycle'] == 10000]['active'].values[0] if len(results) >= 10000 else None
        
        validation = {
            'decay_constant': lambda_decay,
            'threshold': threshold_lambda,
            'passes_threshold': lambda_decay < threshold_lambda,
            'active_fraction_10k': active_10k,
            'estimated_lcoc_factor': lambda_decay / threshold_lambda,
            'meets_lcoc_target': lambda_decay < threshold_lambda
        }
        
        return validation
    
    def plot_degradation(self, 
                        results: pd.DataFrame,
                        save_path: Optional[str] = None):
        """
        Plot state evolution over cycles.
        
        Parameters:
        -----------
        results : pd.DataFrame
            Simulation results
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: State fractions
        ax1.plot(results['cycle'], results['active'], 
                'g-', linewidth=2, label='Active')
        ax1.plot(results['cycle'], results['blocked'], 
                'orange', linewidth=2, label='Blocked')
        ax1.plot(results['cycle'], results['lost'], 
                'r-', linewidth=2, label='Lost')
        
        ax1.set_xlabel('Cycle Number', fontsize=12)
        ax1.set_ylabel('Fraction of Sites', fontsize=12)
        ax1.set_title('Sorbent Degradation Over Cycles', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Log scale for active sites (exponential decay)
        ax2.semilogy(results['cycle'], results['active'], 
                    'g-', linewidth=2, label='Active (log scale)')
        
        # Fit and plot exponential
        lambda_decay = self.calculate_decay_constant(results)
        fit_active = np.exp(-lambda_decay * results['cycle'])
        ax2.semilogy(results['cycle'], fit_active, 
                    'k--', linewidth=2, 
                    label=f'Exp fit: λ={lambda_decay:.2e} cycle⁻¹')
        
        ax2.set_xlabel('Cycle Number', fontsize=12)
        ax2.set_ylabel('Fraction of Active Sites (log)', fontsize=12)
        ax2.set_title('Exponential Decay Analysis', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
