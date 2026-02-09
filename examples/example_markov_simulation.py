"""
Example: Module 3 - Markov Chain Degradation Simulation

This example demonstrates:
1. Setting up the three-state Markov Chain (Active, Blocked, Lost)
2. Defining S-TVSA cycle stages with operating conditions
3. Calculating transition probabilities based on chemical stressors
4. Simulating 1,000 to 10,000 cycles
5. Validating against LCOC targets
6. Visualizing degradation dynamics
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dac_framework.markov_degradation import (
    MarkovDegradation,
    TransitionMatrix,
    StressorParameters,
    CycleConditions,
    CycleStage,
    SiteState
)

def example_basic_simulation():
    """Example 1: Basic Markov Chain simulation."""
    print("="*70)
    print("Example 1: Basic Markov Chain Simulation (1000 cycles)")
    print("="*70)
    
    # Initialize with all sites active
    markov = MarkovDegradation(initial_distribution=[1.0, 0.0, 0.0])
    
    # Simulate 1000 cycles
    results = markov.simulate_cycles(n_cycles=1000, record_interval=10)
    
    print("\nSimulation Results (every 100 cycles):")
    print(results[::10].to_string(index=False))
    
    # Calculate decay constant
    lambda_decay = markov.calculate_decay_constant(results)
    print(f"\nDecay constant (λ): {lambda_decay:.2e} cycle⁻¹")
    
    # Final state after 1000 cycles
    final_state = results.iloc[-1]
    print(f"\nFinal state distribution:")
    print(f"  Active:  {final_state['active']:.4f} ({final_state['active']*100:.2f}%)")
    print(f"  Blocked: {final_state['blocked']:.4f} ({final_state['blocked']*100:.2f}%)")
    print(f"  Lost:    {final_state['lost']:.4f} ({final_state['lost']*100:.2f}%)")
    
    return results


def example_long_term_simulation():
    """Example 2: Long-term simulation (10,000 cycles)."""
    print("\n" + "="*70)
    print("Example 2: Long-Term Simulation (10,000 cycles)")
    print("="*70)
    
    markov = MarkovDegradation()
    
    # Simulate 10,000 cycles (record every 100 cycles)
    print("\nSimulating 10,000 cycles...")
    results = markov.simulate_cycles(n_cycles=10000, record_interval=100)
    
    print(f"\nSimulation completed. Recorded {len(results)} data points.")
    
    # Show key milestones
    milestones = [1000, 2500, 5000, 7500, 10000]
    print("\nState distribution at key milestones:")
    print(f"{'Cycle':>8s} {'Active':>10s} {'Blocked':>10s} {'Lost':>10s}")
    print("-" * 42)
    
    for cycle in milestones:
        state = results[results['cycle'] == cycle].iloc[0]
        print(f"{cycle:>8d} {state['active']:>10.4f} {state['blocked']:>10.4f} {state['lost']:>10.4f}")
    
    # Plot degradation
    markov.plot_degradation(results, save_path='degradation_10k.png')
    
    return results


def example_custom_stressors():
    """Example 3: Custom chemical stressor parameters."""
    print("\n" + "="*70)
    print("Example 3: Custom Chemical Stressor Parameters")
    print("="*70)
    
    # Create custom stressor parameters (harsh conditions)
    harsh_stressors = StressorParameters(
        ea_cn_cleavage=18.0,  # Lower activation energy (more degradation)
        water_block_rate=2e-4,  # Higher water blocking
        oxidation_rate_constant=2e-6,  # Higher oxidation
        thermal_degradation_rate=2e-7  # Higher thermal degradation
    )
    
    # Create benign stressor parameters
    benign_stressors = StressorParameters(
        ea_cn_cleavage=22.0,  # Higher activation energy (less degradation)
        water_block_rate=5e-5,  # Lower water blocking
        oxidation_rate_constant=5e-7,  # Lower oxidation
        thermal_degradation_rate=5e-8  # Lower thermal degradation
    )
    
    # Simulate both scenarios
    markov_harsh = MarkovDegradation(stressors=harsh_stressors)
    markov_benign = MarkovDegradation(stressors=benign_stressors)
    
    print("\nSimulating harsh conditions...")
    results_harsh = markov_harsh.simulate_cycles(n_cycles=10000, record_interval=100)
    
    print("Simulating benign conditions...")
    results_benign = markov_benign.simulate_cycles(n_cycles=10000, record_interval=100)
    
    # Compare final states
    print("\nFinal state comparison (after 10,000 cycles):")
    print(f"{'Condition':>15s} {'Active':>10s} {'Blocked':>10s} {'Lost':>10s}")
    print("-" * 50)
    
    final_harsh = results_harsh.iloc[-1]
    final_benign = results_benign.iloc[-1]
    
    print(f"{'Harsh':>15s} {final_harsh['active']:>10.4f} {final_harsh['blocked']:>10.4f} {final_harsh['lost']:>10.4f}")
    print(f"{'Benign':>15s} {final_benign['active']:>10.4f} {final_benign['blocked']:>10.4f} {final_benign['lost']:>10.4f}")
    
    # Compare decay constants
    lambda_harsh = markov_harsh.calculate_decay_constant(results_harsh)
    lambda_benign = markov_benign.calculate_decay_constant(results_benign)
    
    print(f"\nDecay constants:")
    print(f"  Harsh:  λ = {lambda_harsh:.2e} cycle⁻¹")
    print(f"  Benign: λ = {lambda_benign:.2e} cycle⁻¹")
    print(f"  Ratio:  {lambda_harsh/lambda_benign:.2f}x faster degradation")
    
    return results_harsh, results_benign


def example_lcoc_validation():
    """Example 4: LCOC target validation."""
    print("\n" + "="*70)
    print("Example 4: LCOC Target Validation")
    print("="*70)
    
    markov = MarkovDegradation()
    
    # Simulate 10,000 cycles
    results = markov.simulate_cycles(n_cycles=10000, record_interval=100)
    
    # Validate against LCOC target
    validation = markov.validate_lcoc_target(
        results,
        target_lcoc=100.0,
        threshold_lambda=5e-6
    )
    
    print("\nLCOC Validation Results:")
    print(f"  Target LCOC:                ${validation['threshold']:.2f}/tonne CO₂")
    print(f"  Decay constant (λ):         {validation['decay_constant']:.2e} cycle⁻¹")
    print(f"  Threshold (λ):              {validation['threshold']:.2e} cycle⁻¹")
    print(f"  Passes threshold:           {validation['passes_threshold']}")
    print(f"  Active fraction (10k):      {validation['active_fraction_10k']:.4f}")
    print(f"  LCOC cost factor:           {validation['estimated_lcoc_factor']:.2f}x")
    print(f"  Meets LCOC target:          {validation['meets_lcoc_target']}")
    
    if validation['meets_lcoc_target']:
        print("\n✓ Sorbent meets LCOC target of $100/tonne CO₂")
    else:
        print("\n✗ Sorbent does NOT meet LCOC target")
        print(f"  Estimated LCOC: ${100 * validation['estimated_lcoc_factor']:.2f}/tonne CO₂")
    
    return validation


def example_custom_cycle_stages():
    """Example 5: Custom S-TVSA cycle stages."""
    print("\n" + "="*70)
    print("Example 5: Custom S-TVSA Cycle Stages")
    print("="*70)
    
    # Define custom cycle with extreme conditions
    extreme_cycle = [
        (CycleStage.ADSORPTION, CycleConditions(
            temperature=35, pressure=1013, humidity=0.8, duration=7200,  # Hot, humid
            co2_partial_pressure=0.4
        )),
        (CycleStage.PURGING, CycleConditions(
            temperature=35, pressure=1013, humidity=0.6, duration=600,
            co2_partial_pressure=0.05
        )),
        (CycleStage.HEATING, CycleConditions(
            temperature=110, pressure=300, humidity=0.1, duration=2400,  # High temp
            co2_partial_pressure=0.0
        )),
        (CycleStage.VACUUM, CycleConditions(
            temperature=120, pressure=20, humidity=0.01, duration=1800,  # Deep vacuum
            co2_partial_pressure=0.0
        )),
        (CycleStage.DESORPTION, CycleConditions(
            temperature=120, pressure=20, humidity=0.01, duration=2400,
            co2_partial_pressure=0.0
        )),
        (CycleStage.COOLING, CycleConditions(
            temperature=60, pressure=100, humidity=0.2, duration=1200,
            co2_partial_pressure=0.0
        )),
        (CycleStage.PRESSURIZATION, CycleConditions(
            temperature=35, pressure=1013, humidity=0.5, duration=600,
            co2_partial_pressure=0.2
        )),
    ]
    
    print("\nCustom cycle configuration:")
    print(f"{'Stage':>20s} {'Temp (°C)':>10s} {'P (mbar)':>10s} {'RH':>8s} {'Time (min)':>12s}")
    print("-" * 65)
    
    for stage, conditions in extreme_cycle:
        print(f"{stage.value:>20s} {conditions.temperature:>10.1f} {conditions.pressure:>10.1f} "
              f"{conditions.humidity:>8.2f} {conditions.duration/60:>12.1f}")
    
    # Simulate with custom cycle
    markov = MarkovDegradation()
    print("\nSimulating 5,000 cycles with extreme conditions...")
    results = markov.simulate_cycles(
        n_cycles=5000,
        cycle_stages=extreme_cycle,
        record_interval=50
    )
    
    # Compare with standard cycle
    markov_standard = MarkovDegradation()
    print("Simulating 5,000 cycles with standard conditions...")
    results_standard = markov_standard.simulate_cycles(
        n_cycles=5000,
        record_interval=50
    )
    
    # Compare final states
    print("\nFinal state comparison (after 5,000 cycles):")
    print(f"{'Cycle Type':>15s} {'Active':>10s} {'Blocked':>10s} {'Lost':>10s}")
    print("-" * 50)
    
    final_extreme = results.iloc[-1]
    final_standard = results_standard.iloc[-1]
    
    print(f"{'Extreme':>15s} {final_extreme['active']:>10.4f} {final_extreme['blocked']:>10.4f} {final_extreme['lost']:>10.4f}")
    print(f"{'Standard':>15s} {final_standard['active']:>10.4f} {final_standard['blocked']:>10.4f} {final_standard['lost']:>10.4f}")
    
    return results, results_standard


def example_stage_by_stage_analysis():
    """Example 6: Analyze individual cycle stages."""
    print("\n" + "="*70)
    print("Example 6: Stage-by-Stage Transition Analysis")
    print("="*70)
    
    # Create transition matrix calculator
    trans_calc = TransitionMatrix()
    
    # Define standard cycle stages
    markov = MarkovDegradation()
    cycle_stages = markov.define_cycle_stages()
    
    print("\nTransition matrices for each stage:")
    print("(States: 0=Active, 1=Blocked, 2=Lost)")
    
    for stage, conditions in cycle_stages:
        P = trans_calc.build_transition_matrix(stage, conditions)
        
        print(f"\n{stage.value.upper()}")
        print(f"  Temperature: {conditions.temperature}°C")
        print(f"  Pressure: {conditions.pressure} mbar")
        print(f"  Humidity: {conditions.humidity*100:.1f}%")
        print("  Transition matrix:")
        print(f"    [Active  -> Active, Blocked, Lost] = [{P[0,0]:.4f}, {P[0,1]:.4f}, {P[0,2]:.4f}]")
        print(f"    [Blocked -> Active, Blocked, Lost] = [{P[1,0]:.4f}, {P[1,1]:.4f}, {P[1,2]:.4f}]")
        print(f"    [Lost    -> Active, Blocked, Lost] = [{P[2,0]:.4f}, {P[2,1]:.4f}, {P[2,2]:.4f}]")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MODULE 3: MARKOV CHAIN DEGRADATION SIMULATION")
    print("Examples and Demonstrations")
    print("="*70 + "\n")
    
    # Run examples
    results_1k = example_basic_simulation()
    results_10k = example_long_term_simulation()
    results_harsh, results_benign = example_custom_stressors()
    validation = example_lcoc_validation()
    results_extreme, results_standard = example_custom_cycle_stages()
    example_stage_by_stage_analysis()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("Generated plots: degradation_10k.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
