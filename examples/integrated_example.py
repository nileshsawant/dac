"""
Complete Integration Example: Beyond Isotherms Framework

This example demonstrates the full integration of all four modules:
1. Molecular Physics - QM-based binding energies
2. GNN Surrogate - Performance prediction
3. Markov Degradation - Long-term cycle simulation
4. Geospatial Performance - Climate-integrated analysis

Workflow:
---------
1. Start with molecular-level binding motifs (Module 1)
2. Use those to inform sorbent descriptors
3. Predict performance targets (Module 2 - conceptual, data needed)
4. Simulate degradation over 10,000 cycles (Module 3)
5. Calculate location-specific CRR with degradation (Module 4)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dac_framework import (
    MolecularPhysics,
    MarkovDegradation,
    GeospatialPerformance,
    MERRA2DataLoader,
    SorbentParameters,
    Location,
    PerformanceCalculator,
    ClimateOptimizer
)

def integrated_workflow():
    """
    Complete integrated workflow from molecular physics to geospatial siting.
    """
    
    print("="*80)
    print(" BEYOND ISOTHERMS: INTEGRATED MULTI-SCALE DAC MODELING")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Molecular Physics Analysis
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 1: Molecular Physics & Binding Energy Analysis")
    print("─"*80)
    
    mp = MolecularPhysics(temperature=298.15)
    
    # Calculate binding motif distribution
    motif_dist = mp.calculate_motif_distribution()
    print("\nBinding Motif Distribution at 25°C:")
    print(motif_dist.to_string(index=False))
    
    # Analyze reversibility
    reversible_fraction = motif_dist[motif_dist['Reversible']]['Probability'].sum()
    print(f"\nReversible binding fraction: {reversible_fraction:.4f}")
    print(f"Irreversible binding fraction: {1-reversible_fraction:.4f}")
    
    # Temperature sensitivity
    temps = np.array([298.15, 323.15, 348.15, 373.15])
    print("\nTemperature-dependent equilibrium:")
    for T in temps:
        calc_temp = MolecularPhysics(temperature=T)
        dist_temp = calc_temp.calculate_motif_distribution()
        rev_frac = dist_temp[dist_temp['Reversible']]['Probability'].sum()
        print(f"  {T-273.15:>5.1f}°C: Reversible = {rev_frac:.4f}")
    
    # ================================= (GNN + Climate Optimization)
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 2: Sorbent Design & Performance Parameters")
    print("─"*80)
    
    # ----- MODULE 2 (GNN) PLACEHOLDER -----
    # In a full implementation, this would train a GNN on molecular structures
    # and predict optimal sorbent parameters:
    print("\n[MODULE 2 - GNN Placeholder]")
    print("  In production: Train GNN on molecular dataset")
    print("  → Input: Molecular structures from QM calculations")
    print("  → Output: Predicted CO₂ capacity, H₂O penalty, stability metrics")
    print("  For now: Using informed estimates based on molecular physics")
    
    # GNN would predict these molecular-level properties
    gnn_predictions = {
        'optimized': {'co2_capacity': 2.8, 'h2o_penalty': 0.12, 'temp_coef': 0.012},
        'baseline': {'co2_capacity': 2.2, 'h2o_penalty': 0.20, 'temp_coef': 0.018}
    }
    
    # Start with basic sorbent configurations from GNN predictions
    sorbent_optimized = SorbentParameters(
        co2_capacity_ref=gnn_predictions['optimized']['co2_capacity'],
        working_capacity=2.0,  # mol/kg
        h2o_penalty_factor=gnn_predictions['optimized']['h2o_penalty'],
        temp_coefficient=gnn_predictions['optimized']['temp_coef'],
        humidity_coefficient=1.0,
        regeneration_temp=100.0,  # Will be optimized by ClimateOptimizer
        regeneration_vacuum=60.0,  # mbar
        cycle_time_base=6.0  # Will be optimized by ClimateOptimizer
    )
    
    sorbent_baseline = SorbentParameters(
        co2_capacity_ref=gnn_predictions['baseline']['co2_capacity'],
        working_capacity=1.5,  # mol/kg
        h2o_penalty_factor=gnn_predictions['baseline']['h2o_penalty'],
        temp_coefficient=gnn_predictions['baseline']['temp_coef'],
        humidity_coefficient=1.3,
        regeneration_temp=105.0,  # Will be optimized
        regeneration_vacuum=50.0,  # mbar
        cycle_time_base=6.5  # hours
    )
    
    print("\nInitial sorbent configurations (from GNN):")
    print("\n  OPTIMIZED:")
    print(f"    CO₂ capacity: {sorbent_optimized.co2_capacity_ref} mol/kg")
    print(f"    H₂O penalty:  {sorbent_optimized.h2o_penalty_factor}")
    print(f"    Regen temp:   {sorbent_optimized.regeneration_temp}°C (pre-optimization)")
    print(f"    Cycle time:   {sorbent_optimized.cycle_time_base} hours (pre-optimization)")
    
    print("\n  BASELINE:")
    print(f"    CO₂ capacity: {sorbent_baseline.co2_capacity_ref} mol/kg")
    print(f"    H₂O penalty:  {sorbent_baseline.h2o_penalty_factor}")
    print(f"    Regen temp:   {sorbent_baseline.regeneration_temp}°C (pre-optimization)")
    print(f"    Cycle time:   {sorbent_baseline.cycle_time_base} hours (pre-optimization)")
    
    # ----- CLIMATE OPTIMIZATION -----
    print("\n[MODULE 4 - Climate Optimization]")
    print("  Optimizing operating parameters for Phoenix climate...")
    
    # Generate sample weather data for optimization (using Phoenix/arid climate)
    loader = MERRA2DataLoader()
    phoenix_weather = loader.generate_synthetic_data(n_hours=2000, location='arid')
    
    # Optimize both sorbents for Phoenix climate
    optimizer_opt = ClimateOptimizer(sorbent_optimized, phoenix_weather)
    result_opt = optimizer_opt.optimize_pso(
        bounds=[(4.0, 8.0), (85.0, 105.0)],  # [cycle_time, regen_temp]
        swarmsize=20,
        maxiter=50
    )
    
    optimizer_base = ClimateOptimizer(sorbent_baseline, phoenix_weather)
    result_base = optimizer_base.optimize_pso(
        bounds=[(4.0, 8.0), (90.0, 110.0)],
        swarmsize=20,
        maxiter=50
    )
    
    print(f"\n  OPTIMIZED sorbent - Climate-optimized parameters:")
    print(f"    Optimal cycle time: {result_opt['optimal_cycle_time']:.2f} hours")
    print(f"    Optimal regen temp: {result_opt['optimal_regen_temp']:.1f}°C")
    print(f"    Max CRR achieved:   {result_opt['max_crr']:.4f} mol CO₂/kg·h")
    
    print(f"\n  BASELINE sorbent - Climate-optimized parameters:")
    print(f"    Optimal cycle time: {result_base['optimal_cycle_time']:.2f} hours")
    print(f"    Optimal regen temp: {result_base['optimal_regen_temp']:.1f}°C")
    print(f"    Max CRR achieved:   {result_base['max_crr']:.4f} mol CO₂/kg·h")
    
    # Update sorbents with optimized parameters
    sorbent_optimized.cycle_time_base = result_opt['optimal_cycle_time']
    sorbent_optimized.regeneration_temp = result_opt['optimal_regen_temp']
    sorbent_baseline.cycle_time_base = result_base['optimal_cycle_time']
    sorbent_baseline.regeneration_temp = result_base['optimal_regen_temp']
    
    # ========================================================================
    # STEP 3: Long-Term Degradation Analysis
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 3: Markov Chain Degradation Simulation (10,000 cycles)")
    print("─"*80)
    
    # Simulate degradation for both sorbents
    print("\nSimulating optimized sorbent degradation...")
    markov_opt = MarkovDegradation()
    deg_opt = markov_opt.simulate_cycles(n_cycles=10000, record_interval=100)
    
    print("Simulating baseline sorbent degradation...")
    # Baseline has moderately worse degradation (more visible in plots)
    from dac_framework.markov_degradation import StressorParameters
    baseline_stressors = StressorParameters(
        ea_cn_cleavage=50.0,  # Lower than optimized (60) but not catastrophic
        water_block_rate=1.5e-4,  # 1.9x higher water blocking
        water_recovery_rate=0.010,  # Slower recovery (vs 0.015 default)
        oxidation_rate_constant=1.5e-7,  # 3x higher oxidation
        thermal_degradation_rate=2.0e-8  # 2x higher thermal degradation
    )
    markov_base = MarkovDegradation(stressors=baseline_stressors)
    deg_base = markov_base.simulate_cycles(n_cycles=10000, record_interval=100)
    
    # Validate against LCOC
    validation_opt = markov_opt.validate_lcoc_target(deg_opt)
    validation_base = markov_base.validate_lcoc_target(deg_base)
    
    print("\nDegradation Results (10,000 cycles):")
    print(f"\n  OPTIMIZED:")
    print(f"    Active sites remaining: {deg_opt.iloc[-1]['active']:.4f}")
    print(f"    Decay constant (λ):     {validation_opt['decay_constant']:.2e} cycle⁻¹")
    print(f"    Meets LCOC target:      {validation_opt['meets_lcoc_target']}")
    
    print(f"\n  BASELINE:")
    print(f"    Active sites remaining: {deg_base.iloc[-1]['active']:.4f}")
    print(f"    Decay constant (λ):     {validation_base['decay_constant']:.2e} cycle⁻¹")
    print(f"    Meets LCOC target:      {validation_base['meets_lcoc_target']}")
    
    # ========================================================================
    # STEP 4: Geospatial Analysis with Degradation
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 4: Geospatial Performance Analysis (Climate Integration)")
    print("─"*80)
    
    # Setup locations
    geo = GeospatialPerformance()
    
    locations = [
        Location("Phoenix, AZ (Arid)", 33.45, -112.07),
        Location("Seattle, WA (Temperate)", 47.61, -122.33),
        Location("Reykjavik, Iceland (Cold)", 64.15, -21.94),
    ]
    
    climate_map = {
        "Phoenix, AZ (Arid)": "arid",
        "Seattle, WA (Temperate)": "temperate",
        "Reykjavik, Iceland (Cold)": "polar",
    }
    
    # Generate weather data
    loader = MERRA2DataLoader()
    
    print("\nGenerating weather data for locations...")
    for location in locations:
        climate = climate_map[location.name]
        weather = loader.generate_synthetic_data(n_hours=8760, location=climate)
        geo.add_location(location, weather)
        print(f"  {location.name}: {len(weather)} hours")
    
    # Compare optimized vs baseline across locations
    print("\nCalculating performance for OPTIMIZED sorbent...")
    comparison_opt = geo.compare_locations(sorbent_optimized)
    
    print("\nCalculating performance for BASELINE sorbent...")
    comparison_base = geo.compare_locations(sorbent_baseline)
    
    print("\nPerformance Comparison - OPTIMIZED Sorbent:")
    print(comparison_opt.to_string(index=False))
    
    print("\nPerformance Comparison - BASELINE Sorbent:")
    print(comparison_base.to_string(index=False))
    
    # ========================================================================
    # STEP 5: Integrated Performance with Degradation
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 5: Integrated Analysis (Performance + Degradation)")
    print("─"*80)
    
    # Calculate lifetime performance including degradation
    
    def calculate_lifetime_crr(location_name, sorbent, degradation_data, 
                              geo_analyzer, n_years=3):
        """Calculate lifetime-averaged CRR accounting for degradation."""
        
        # Get hourly CRR for first year
        crr_year1 = geo_analyzer.calculate_crr(location_name, sorbent, n_years=1)
        avg_crr_year1 = crr_year1['crr'].mean()
        
        # Estimate cycle numbers
        avg_cycle_time = crr_year1['cycle_time'].mean()
        cycles_per_year = 8760 / avg_cycle_time
        
        # Degradation factors over lifetime
        year_crrs = []
        for year in range(n_years):
            cycle_start = int(year * cycles_per_year)
            cycle_end = int((year + 1) * cycles_per_year)
            
            # Get degradation factor at mid-year
            cycle_mid = (cycle_start + cycle_end) // 2
            
            if cycle_mid < len(degradation_data):
                deg_factor = degradation_data.iloc[cycle_mid // 100]['active']
            else:
                # Extrapolate
                lambda_decay = markov_opt.calculate_decay_constant(degradation_data)
                deg_factor = np.exp(-lambda_decay * cycle_mid)
            
            year_crr = avg_crr_year1 * deg_factor
            year_crrs.append(year_crr)
        
        lifetime_avg_crr = np.mean(year_crrs)
        return lifetime_avg_crr, year_crrs
    
    print("\n3-Year Lifetime Performance Analysis:")
    print(f"\n{'Location':<30s} {'Optimized':>12s} {'Baseline':>12s} {'Improvement':>12s}")
    print("─" * 70)
    
    for location in locations:
        lifetime_opt, years_opt = calculate_lifetime_crr(
            location.name, sorbent_optimized, deg_opt, geo
        )
        lifetime_base, years_base = calculate_lifetime_crr(
            location.name, sorbent_baseline, deg_base, geo
        )
        
        improvement = ((lifetime_opt - lifetime_base) / lifetime_base) * 100
        
        print(f"{location.name:<30s} {lifetime_opt:>12.4f} {lifetime_base:>12.4f} "
              f"{improvement:>11.1f}%")
    
    # ========================================================================
    # STEP 6: Visualization
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 6: Generating Integrated Visualizations")
    print("─"*80)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Degradation comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(deg_opt['cycle'], deg_opt['active'], 'g-', linewidth=2, label='Optimized')
    ax1.plot(deg_base['cycle'], deg_base['active'], 'r--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Active Site Fraction')
    ax1.set_title('Degradation Over 10,000 Cycles', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Performance by location (Optimized)
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(comparison_opt['location'], comparison_opt['avg_crr'], color='green', alpha=0.7)
    ax2.set_xlabel('Average CRR (mol CO₂/kg·h)')
    ax2.set_title('Optimized Sorbent Performance', fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    
    # Plot 3: Performance by location (Baseline)
    ax3 = plt.subplot(2, 3, 3)
    ax3.barh(comparison_base['location'], comparison_base['avg_crr'], color='red', alpha=0.7)
    ax3.set_xlabel('Average CRR (mol CO₂/kg·h)')
    ax3.set_title('Baseline Sorbent Performance', fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # Plot 4: Energy efficiency comparison
    ax4 = plt.subplot(2, 3, 4)
    width = 0.35
    x = np.arange(len(locations))
    ax4.bar(x - width/2, comparison_opt['avg_energy'], width, label='Optimized', color='green', alpha=0.7)
    ax4.bar(x + width/2, comparison_base['avg_energy'], width, label='Baseline', color='red', alpha=0.7)
    ax4.set_ylabel('Average Energy (kWh/tonne CO₂)')
    ax4.set_title('Energy Requirements', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([loc.name.split(',')[0] for loc in locations], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Plot 5: Capacity factor comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(x - width/2, comparison_opt['capacity_factor'], width, label='Optimized', color='green', alpha=0.7)
    ax5.bar(x + width/2, comparison_base['capacity_factor'], width, label='Baseline', color='red', alpha=0.7)
    ax5.set_ylabel('Capacity Factor (%)')
    ax5.set_title('Operational Efficiency', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([loc.name.split(',')[0] for loc in locations], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # Plot 6: Overall improvement
    ax6 = plt.subplot(2, 3, 6)
    improvements = []
    for i, location in enumerate(locations):
        opt_crr = comparison_opt.iloc[i]['avg_crr']
        base_crr = comparison_base.iloc[i]['avg_crr']
        improvement = ((opt_crr - base_crr) / base_crr) * 100
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax6.barh([loc.name.split(',')[0] for loc in locations], improvements, color=colors, alpha=0.7)
    ax6.set_xlabel('CRR Improvement (%)')
    ax6.set_title('Optimized vs Baseline', fontweight='bold')
    ax6.axvline(0, color='black', linewidth=0.5)
    ax6.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('integrated_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nIntegrated visualization saved as 'integrated_analysis.png'")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" SUMMARY: INTEGRATED MULTI-SCALE ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nKey Findings:")
    print("  1. Molecular-level binding analysis shows {:.1f}% reversible binding".format(
        reversible_fraction * 100))
    print("  2. Optimized sorbent maintains {:.1f}% active sites after 10,000 cycles".format(
        deg_opt.iloc[-1]['active'] * 100))
    print("  3. Baseline sorbent maintains {:.1f}% active sites after 10,000 cycles".format(
        deg_base.iloc[-1]['active'] * 100))
    print("  4. Best location: {} with {:.4f} mol CO₂/kg·h (optimized)".format(
        comparison_opt.iloc[0]['location'], comparison_opt.iloc[0]['avg_crr']))
    print("  5. Average improvement: {:.1f}% across all locations".format(
        np.mean(improvements)))
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    integrated_workflow()
