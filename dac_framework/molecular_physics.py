"""
Module 1: Molecular Physics & Transition Logic
==============================================

This module converts quantum mechanical binding energies into probabilistic 
event statistics using Boltzmann distributions.

Key Features:
- 6-membered concerted mechanism for CO2 capture
- Parsing of Gaussian/VASP outputs
- Energy distribution histograms
- State occupancy calculations based on DFT binding energies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
from scipy import stats


# Physical constants
R = 8.314e-3  # Gas constant in kcal/(mol·K)
KB = 1.381e-23  # Boltzmann constant in J/K
KCAL_TO_J = 4184  # Conversion factor


@dataclass
class BindingMotif:
    """Represents a CO2 binding motif with its thermodynamic properties."""
    name: str
    delta_g: float  # Free energy in kcal/mol
    delta_h: Optional[float] = None  # Enthalpy in kcal/mol
    delta_s: Optional[float] = None  # Entropy in cal/(mol·K)
    is_reversible: bool = True
    
    def __repr__(self):
        return f"{self.name}: ΔG = {self.delta_g:.2f} kcal/mol"


class BoltzmannCalculator:
    """
    Calculate Boltzmann-weighted state probabilities from QM energy data.
    """
    
    def __init__(self, temperature: float = 298.15):
        """
        Initialize calculator.
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin (default: 298.15 K = 25°C)
        """
        self.temperature = temperature
        self.beta = 1.0 / (R * temperature)  # 1/(RT) in mol/kcal
        
    def calculate_state_probability(self, delta_g: float, 
                                    reference_g: float = 0.0) -> float:
        """
        Calculate probability of a state using Boltzmann distribution.
        
        P_i = exp(-ΔG_i/RT) / Z
        
        Parameters:
        -----------
        delta_g : float
            Free energy of state in kcal/mol
        reference_g : float
            Reference free energy (default: 0.0)
            
        Returns:
        --------
        float
            Unnormalized probability (needs partition function for normalization)
        """
        relative_g = delta_g - reference_g
        return np.exp(-self.beta * relative_g)
    
    def calculate_state_distribution(self, 
                                    energies: List[float],
                                    degeneracies: Optional[List[int]] = None) -> np.ndarray:
        """
        Calculate normalized probability distribution across states.
        
        Parameters:
        -----------
        energies : List[float]
            List of free energies for each state (kcal/mol)
        degeneracies : List[int], optional
            Degeneracy of each state (default: all 1)
            
        Returns:
        --------
        np.ndarray
            Normalized probability distribution
        """
        energies = np.array(energies)
        if degeneracies is None:
            degeneracies = np.ones_like(energies)
        else:
            degeneracies = np.array(degeneracies)
        
        # Calculate unnormalized probabilities
        probs = degeneracies * np.exp(-self.beta * energies)
        
        # Normalize (partition function)
        Z = np.sum(probs)
        
        return probs / Z
    
    def calculate_equilibrium_constant(self, delta_g: float) -> float:
        """
        Calculate equilibrium constant from free energy.
        
        K_eq = exp(-ΔG/RT)
        
        Parameters:
        -----------
        delta_g : float
            Free energy of reaction in kcal/mol
            
        Returns:
        --------
        float
            Equilibrium constant
        """
        return np.exp(-self.beta * delta_g)
    
    def calculate_rate_constant(self, ea: float, pre_exponential: float = 1e13) -> float:
        """
        Calculate rate constant using Arrhenius equation.
        
        k = A * exp(-E_a/RT)
        
        Parameters:
        -----------
        ea : float
            Activation energy in kcal/mol
        pre_exponential : float
            Pre-exponential factor A (default: 1e13 s^-1)
            
        Returns:
        --------
        float
            Rate constant in s^-1
        """
        return pre_exponential * np.exp(-self.beta * ea)


class MolecularPhysics:
    """
    Main class for molecular physics calculations and QM data processing.
    """
    
    # Default binding motifs based on DFT analysis
    DEFAULT_MOTIFS = {
        'alkylammonium_carbamate': BindingMotif(
            name='Alkylammonium Carbamate',
            delta_g=-26.5,
            is_reversible=True
        ),
        'water_stabilized': BindingMotif(
            name='Water-Stabilized Complex',
            delta_g=-22.0,
            is_reversible=True
        ),
        'irreversible_urea': BindingMotif(
            name='Irreversible Urea',
            delta_g=-35.0,
            is_reversible=False
        ),
    }
    
    def __init__(self, temperature: float = 298.15):
        """
        Initialize molecular physics module.
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        """
        self.temperature = temperature
        self.calculator = BoltzmannCalculator(temperature)
        self.motifs = self.DEFAULT_MOTIFS.copy()
        
    def add_motif(self, key: str, motif: BindingMotif):
        """Add a new binding motif to the collection."""
        self.motifs[key] = motif
        
    def calculate_motif_distribution(self) -> pd.DataFrame:
        """
        Calculate probability distribution across all binding motifs.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with motif names, energies, and probabilities
        """
        motif_names = []
        energies = []
        reversible = []
        
        for key, motif in self.motifs.items():
            motif_names.append(motif.name)
            energies.append(motif.delta_g)
            reversible.append(motif.is_reversible)
        
        probs = self.calculator.calculate_state_distribution(energies)
        
        return pd.DataFrame({
            'Motif': motif_names,
            'ΔG (kcal/mol)': energies,
            'Probability': probs,
            'Reversible': reversible
        })
    
    def parse_gaussian_output(self, filepath: str) -> Dict:
        """
        Parse Gaussian output file to extract energy data.
        
        Parameters:
        -----------
        filepath : str
            Path to Gaussian .log file
            
        Returns:
        --------
        Dict
            Dictionary containing extracted energies and properties
        """
        results = {
            'scf_energy': None,
            'zero_point_energy': None,
            'thermal_correction': None,
            'gibbs_free_energy': None,
        }
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Extract SCF energy
            scf_match = re.search(r'SCF Done.*?=\s+([-\d.]+)', content)
            if scf_match:
                results['scf_energy'] = float(scf_match.group(1))
            
            # Extract thermochemistry data
            zpe_match = re.search(r'Zero-point correction=\s+([-\d.]+)', content)
            if zpe_match:
                results['zero_point_energy'] = float(zpe_match.group(1))
            
            thermal_match = re.search(r'Thermal correction to Gibbs Free Energy=\s+([-\d.]+)', content)
            if thermal_match:
                results['thermal_correction'] = float(thermal_match.group(1))
            
            gibbs_match = re.search(r'Sum of electronic and thermal Free Energies=\s+([-\d.]+)', content)
            if gibbs_match:
                results['gibbs_free_energy'] = float(gibbs_match.group(1))
                
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
        
        return results
    
    def parse_vasp_output(self, outcar_path: str) -> Dict:
        """
        Parse VASP OUTCAR file to extract energy data.
        
        Parameters:
        -----------
        outcar_path : str
            Path to VASP OUTCAR file
            
        Returns:
        --------
        Dict
            Dictionary containing extracted energies
        """
        results = {
            'total_energy': None,
            'energy_without_entropy': None,
            'free_energy': None,
        }
        
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            
            # Extract energies from last ionic step
            for line in reversed(lines):
                if 'free energy    TOTEN' in line and results['total_energy'] is None:
                    results['total_energy'] = float(line.split()[-2])
                elif 'energy  without entropy' in line and results['energy_without_entropy'] is None:
                    results['energy_without_entropy'] = float(line.split()[-1])
                    
            # For VASP, free energy is approximated by energy without entropy
            results['free_energy'] = results['energy_without_entropy']
            
        except FileNotFoundError:
            print(f"Error: File {outcar_path} not found")
        
        return results
    
    def calculate_binding_energy(self, 
                                 product_energy: float,
                                 reactant_energies: List[float]) -> float:
        """
        Calculate binding energy from product and reactant energies.
        
        ΔE = E_product - Σ(E_reactants)
        
        Parameters:
        -----------
        product_energy : float
            Total energy of product complex
        reactant_energies : List[float]
            List of energies for each reactant
            
        Returns:
        --------
        float
            Binding energy (negative = favorable)
        """
        return product_energy - sum(reactant_energies)
    
    def generate_energy_histogram(self, 
                                  energies: List[float],
                                  title: str = "Energy Distribution",
                                  bins: int = 30,
                                  save_path: Optional[str] = None):
        """
        Generate histogram of energy distribution.
        
        Parameters:
        -----------
        energies : List[float]
            List of energy values
        title : str
            Plot title
        bins : int
            Number of histogram bins
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins_edges, patches = ax.hist(energies, bins=bins, 
                                         edgecolor='black', alpha=0.7)
        
        # Add statistics
        mean_e = np.mean(energies)
        std_e = np.std(energies)
        
        ax.axvline(mean_e, color='r', linestyle='--', 
                   label=f'Mean: {mean_e:.2f} kcal/mol')
        ax.axvline(mean_e + std_e, color='g', linestyle='--', 
                   label=f'±1σ: {std_e:.2f} kcal/mol')
        ax.axvline(mean_e - std_e, color='g', linestyle='--')
        
        ax.set_xlabel('Energy (kcal/mol)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_rdf(self, 
                     distances: np.ndarray,
                     r_max: float = 10.0,
                     n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Radial Distribution Function (RDF) from distance data.
        
        Parameters:
        -----------
        distances : np.ndarray
            Array of distances (e.g., N-C for CO2, O-H for H2O)
        r_max : float
            Maximum distance for RDF calculation
        n_bins : int
            Number of bins for histogram
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (r, g(r)) where r is distance and g(r) is RDF
        """
        # Create histogram of distances
        hist, bin_edges = np.histogram(distances, bins=n_bins, 
                                       range=(0, r_max), density=False)
        
        # Calculate bin centers
        r = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate shell volumes
        dr = bin_edges[1] - bin_edges[0]
        shell_volumes = 4 * np.pi * r**2 * dr
        
        # Normalize by shell volume and number density
        n_atoms = len(distances)
        avg_density = n_atoms / (4/3 * np.pi * r_max**3)
        
        g_r = hist / (shell_volumes * avg_density * n_atoms)
        
        return r, g_r
    
    def plot_rdf(self,
                 r: np.ndarray,
                 g_r: np.ndarray,
                 title: str = "Radial Distribution Function",
                 xlabel: str = "r (Å)",
                 save_path: Optional[str] = None):
        """
        Plot Radial Distribution Function.
        
        Parameters:
        -----------
        r : np.ndarray
            Distance array
        g_r : np.ndarray
            RDF values
        title : str
            Plot title
        xlabel : str
            X-axis label
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(r, g_r, 'b-', linewidth=2)
        ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Bulk density')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def six_membered_mechanism_probability(self,
                                          temperature_range: np.ndarray,
                                          ea_forward: float = 15.0,
                                          ea_reverse: float = 41.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward and reverse rate constants for 6-membered concerted mechanism.
        
        Parameters:
        -----------
        temperature_range : np.ndarray
            Array of temperatures (K)
        ea_forward : float
            Forward activation energy (kcal/mol)
        ea_reverse : float
            Reverse activation energy (kcal/mol), default = 26.5 + 15.0
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (k_forward, k_reverse) rate constants
        """
        k_forward = []
        k_reverse = []
        
        for T in temperature_range:
            calc = BoltzmannCalculator(T)
            k_f = calc.calculate_rate_constant(ea_forward)
            k_r = calc.calculate_rate_constant(ea_reverse)
            k_forward.append(k_f)
            k_reverse.append(k_r)
        
        return np.array(k_forward), np.array(k_reverse)


# Convenience functions
def create_motif_from_dft(name: str, 
                          scf_product: float,
                          scf_reactants: List[float],
                          thermal_product: float,
                          thermal_reactants: List[float],
                          is_reversible: bool = True) -> BindingMotif:
    """
    Create a BindingMotif from DFT calculation results.
    
    Parameters:
    -----------
    name : str
        Name of the binding motif
    scf_product : float
        SCF energy of product (Hartree)
    scf_reactants : List[float]
        List of SCF energies for reactants (Hartree)
    thermal_product : float
        Thermal correction to Gibbs free energy for product (Hartree)
    thermal_reactants : List[float]
        Thermal corrections for reactants (Hartree)
    is_reversible : bool
        Whether the binding is reversible
        
    Returns:
    --------
    BindingMotif
        Binding motif object with calculated ΔG
    """
    # Convert Hartree to kcal/mol (1 Hartree = 627.509 kcal/mol)
    HARTREE_TO_KCAL = 627.509
    
    g_product = (scf_product + thermal_product) * HARTREE_TO_KCAL
    g_reactants = sum((e + t) * HARTREE_TO_KCAL 
                     for e, t in zip(scf_reactants, thermal_reactants))
    
    delta_g = g_product - g_reactants
    
    return BindingMotif(name=name, delta_g=delta_g, is_reversible=is_reversible)
