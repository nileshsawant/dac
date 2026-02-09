"""
Example: Module 1 - Molecular Physics & Transition Logic

This example demonstrates:
1. Calculating Boltzmann-weighted state probabilities
2. Parsing QM output files (Gaussian/VASP)
3. Generating energy distribution histograms
4. Calculating Radial Distribution Functions (RDF)
5. Analyzing 6-membered concerted mechanism
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dac_framework.molecular_physics import (
    MolecularPhysics, 
    BoltzmannCalculator,
    BindingMotif,
    create_motif_from_dft
)

def example_boltzmann_probabilities():
    """Example 1: Calculate Boltzmann probabilities for binding motifs."""
    print("="*70)
    print("Example 1: Boltzmann-Weighted State Probabilities")
    print("="*70)
    
    # Initialize molecular physics module at 298.15 K (25°C)
    mp = MolecularPhysics(temperature=298.15)
    
    # Calculate probability distribution across default motifs
    distribution = mp.calculate_motif_distribution()
    
    print("\nBinding Motif Distribution:")
    print(distribution.to_string(index=False))
    
    # Calculate equilibrium constants
    print("\n\nEquilibrium Constants:")
    for key, motif in mp.motifs.items():
        K_eq = mp.calculator.calculate_equilibrium_constant(motif.delta_g)
        print(f"{motif.name:30s}: K_eq = {K_eq:.2e}")
    
    return distribution


def example_temperature_dependence():
    """Example 2: Temperature dependence of state probabilities."""
    print("\n" + "="*70)
    print("Example 2: Temperature Dependence")
    print("="*70)
    
    temperatures = np.array([273.15, 298.15, 323.15, 348.15, 373.15])  # 0, 25, 50, 75, 100°C
    
    # Alkylammonium carbamate binding energy
    delta_g = -26.5  # kcal/mol
    
    print("\nProbability of Alkylammonium Carbamate Formation:")
    print(f"{'Temperature (°C)':>15s} {'Temperature (K)':>15s} {'Probability':>15s} {'K_eq':>15s}")
    print("-" * 62)
    
    for T in temperatures:
        calc = BoltzmannCalculator(T)
        K_eq = calc.calculate_equilibrium_constant(delta_g)
        # Normalize against unbound state (ΔG = 0)
        prob = K_eq / (1 + K_eq)
        
        print(f"{T-273.15:>15.1f} {T:>15.2f} {prob:>15.4f} {K_eq:>15.2e}")


def example_energy_histogram():
    """Example 3: Generate energy distribution histogram."""
    print("\n" + "="*70)
    print("Example 3: Energy Distribution Analysis")
    print("="*70)
    
    mp = MolecularPhysics()
    
    # Generate synthetic energy data (e.g., from multiple DFT calculations)
    # Simulating Alkylammonium Carbamate formation energies with some spread
    np.random.seed(42)
    n_samples = 100
    mean_energy = -26.5  # kcal/mol
    std_energy = 1.5  # kcal/mol
    
    energies = np.random.normal(mean_energy, std_energy, n_samples)
    
    print(f"\nGenerated {n_samples} binding energy samples")
    print(f"Mean: {np.mean(energies):.2f} kcal/mol")
    print(f"Std:  {np.std(energies):.2f} kcal/mol")
    print(f"Range: [{np.min(energies):.2f}, {np.max(energies):.2f}] kcal/mol")
    
    # Generate histogram
    mp.generate_energy_histogram(
        energies,
        title="CO2-Amine Binding Energy Distribution",
        bins=20,
        save_path="energy_distribution.png"
    )
    
    return energies


def example_rdf_analysis():
    """Example 4: Radial Distribution Function analysis."""
    print("\n" + "="*70)
    print("Example 4: Radial Distribution Function (RDF)")
    print("="*70)
    
    mp = MolecularPhysics()
    
    # Generate synthetic N-C (CO2) distance data from MD simulation
    # Typical first shell peak around 1.45 Å for carbamate
    np.random.seed(42)
    
    # First shell (carbamate)
    first_shell = np.random.normal(1.45, 0.05, 500)
    # Second shell
    second_shell = np.random.normal(2.8, 0.15, 300)
    # Bulk
    bulk_distances = np.random.uniform(3.5, 10.0, 1000)
    
    all_distances = np.concatenate([first_shell, second_shell, bulk_distances])
    
    print(f"\nAnalyzing {len(all_distances)} N-C distance samples")
    
    # Calculate RDF
    r, g_r = mp.calculate_rdf(all_distances, r_max=10.0, n_bins=100)
    
    # Find peaks
    peaks_idx = np.where((g_r[1:-1] > g_r[:-2]) & (g_r[1:-1] > g_r[2:]))[0] + 1
    peaks = [(r[i], g_r[i]) for i in peaks_idx if g_r[i] > 1.5]
    
    print("\nRDF Peaks (coordination shells):")
    for i, (r_peak, g_peak) in enumerate(peaks[:3], 1):
        print(f"  Shell {i}: r = {r_peak:.2f} Å, g(r) = {g_peak:.2f}")
    
    # Plot RDF
    mp.plot_rdf(
        r, g_r,
        title="N-C(CO₂) Radial Distribution Function",
        xlabel="r (Å)",
        save_path="rdf_analysis.png"
    )
    
    return r, g_r


def example_six_membered_mechanism():
    """Example 5: Six-membered concerted mechanism analysis."""
    print("\n" + "="*70)
    print("Example 5: Six-Membered Concerted Mechanism")
    print("="*70)
    
    mp = MolecularPhysics()
    
    # Temperature range for analysis
    temp_range = np.linspace(273.15, 373.15, 20)  # 0 to 100°C
    
    # Activation energies
    ea_forward = 15.0  # kcal/mol
    ea_reverse = 41.5  # kcal/mol (15.0 + 26.5)
    
    print(f"\nForward activation energy:  {ea_forward} kcal/mol")
    print(f"Reverse activation energy:  {ea_reverse} kcal/mol")
    print(f"Reaction free energy:       {ea_reverse - ea_forward} kcal/mol")
    
    # Calculate rate constants
    k_f, k_r = mp.six_membered_mechanism_probability(
        temp_range, ea_forward, ea_reverse
    )
    
    # Print results at key temperatures
    print(f"\n{'Temperature (°C)':>15s} {'k_forward (s⁻¹)':>18s} {'k_reverse (s⁻¹)':>18s} {'K_eq':>12s}")
    print("-" * 65)
    
    for i in [0, 5, 10, 15, 19]:  # Select temperatures
        T_celsius = temp_range[i] - 273.15
        K_eq = k_f[i] / k_r[i]
        print(f"{T_celsius:>15.1f} {k_f[i]:>18.2e} {k_r[i]:>18.2e} {K_eq:>12.2e}")
    
    # Plot rate constants vs temperature
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Rate constants
    ax1.semilogy(temp_range - 273.15, k_f, 'b-', linewidth=2, label='Forward')
    ax1.semilogy(temp_range - 273.15, k_r, 'r-', linewidth=2, label='Reverse')
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Rate Constant (s⁻¹)', fontsize=12)
    ax1.set_title('Arrhenius Kinetics', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Equilibrium constant
    K_eq_array = k_f / k_r
    ax2.semilogy(temp_range - 273.15, K_eq_array, 'g-', linewidth=2)
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Equilibrium Constant', fontsize=12)
    ax2.set_title('Temperature Dependence of K_eq', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mechanism_kinetics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'mechanism_kinetics.png'")


def example_dft_motif_creation():
    """Example 6: Create binding motif from DFT calculations."""
    print("\n" + "="*70)
    print("Example 6: Creating Motif from DFT Data")
    print("="*70)
    
    # Example DFT data (in Hartree)
    # Product: Alkylammonium carbamate complex
    scf_product = -500.1234
    thermal_product = 0.0567
    
    # Reactants: Amine + CO2
    scf_amine = -312.4567
    thermal_amine = 0.0432
    scf_co2 = -187.6543
    thermal_co2 = 0.0045
    
    # Create motif
    motif = create_motif_from_dft(
        name="Alkylammonium Carbamate (DFT)",
        scf_product=scf_product,
        scf_reactants=[scf_amine, scf_co2],
        thermal_product=thermal_product,
        thermal_reactants=[thermal_amine, thermal_co2],
        is_reversible=True
    )
    
    print(f"\nCreated binding motif from DFT:")
    print(motif)
    
    # Add to molecular physics instance
    mp = MolecularPhysics()
    mp.add_motif('dft_carbamate', motif)
    
    distribution = mp.calculate_motif_distribution()
    print("\nUpdated motif distribution:")
    print(distribution.to_string(index=False))


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MODULE 1: MOLECULAR PHYSICS & TRANSITION LOGIC")
    print("Examples and Demonstrations")
    print("="*70 + "\n")
    
    # Run examples
    example_boltzmann_probabilities()
    example_temperature_dependence()
    example_energy_histogram()
    example_rdf_analysis()
    example_six_membered_mechanism()
    example_dft_motif_creation()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
