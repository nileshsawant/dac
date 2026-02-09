"""
Example: Module 4 - Geospatial Performance & Siting

This example demonstrates:
1. Loading and processing MERRA-2 weather data
2. Calculating location-specific Carbon Removal Rate (CRR)
3. Dynamic cycle adjustment based on ambient conditions
4. Optimization using Particle Swarm Optimization
5. Multi-location performance comparison
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dac_framework.geospatial_performance import (
    GeospatialPerformance,
    MERRA2DataLoader,
    PerformanceCalculator,
    ClimateOptimizer,
    SorbentParameters,
    Location
)

def example_weather_data_generation():
    """Example 1: Generate synthetic weather data for different climates."""
    print("="*70)
    print("Example 1: Weather Data Generation")
    print("="*70)
    
    loader = MERRA2DataLoader()
    
    # Generate data for different climate zones
    climates = ["temperate", "tropical", "arid", "polar"]
    
    weather_datasets = {}
    
    for climate in climates:
        print(f"\nGenerating synthetic data for {climate} climate...")
        df = loader.generate_synthetic_data(n_hours=8760, location=climate)
        weather_datasets[climate] = df
        
        # Show statistics
        print(f"  Temperature: {df['temperature'].mean():.1f}°C "
              f"(range: {df['temperature'].min():.1f} to {df['temperature'].max():.1f})")
        print(f"  Humidity:    {df['relative_humidity'].mean():.2f} "
              f"(range: {df['relative_humidity'].min():.2f} to {df['relative_humidity'].max():.2f})")
        print(f"  Pressure:    {df['pressure'].mean():.1f} mbar")
    
    return weather_datasets


def example_crr_calculation():
    """Example 2: Calculate Carbon Removal Rate (CRR)."""
    print("\n" + "="*70)
    print("Example 2: Carbon Removal Rate Calculation")
    print("="*70)
    
    # Define sorbent parameters
    sorbent = SorbentParameters(
        co2_capacity_ref=2.5,  # mol/kg at reference conditions
        working_capacity=1.8,  # mol/kg
        h2o_penalty_factor=0.15,
        temp_coefficient=0.015,  # 1.5% per °C
        humidity_coefficient=1.2,
        regeneration_temp=100.0,  # °C
        regeneration_vacuum=50.0,  # mbar
        cycle_time_base=6.0  # hours
    )
    
    # Generate weather data
    loader = MERRA2DataLoader()
    weather = loader.generate_synthetic_data(n_hours=168, location="temperate")  # 1 week
    
    print("\nSorbent parameters:")
    print(f"  Reference CO₂ capacity: {sorbent.co2_capacity_ref} mol/kg")
    print(f"  Working capacity: {sorbent.working_capacity} mol/kg")
    print(f"  Regeneration temp: {sorbent.regeneration_temp}°C")
    print(f"  Base cycle time: {sorbent.cycle_time_base} hours")
    
    # Calculate performance
    calculator = PerformanceCalculator(sorbent)
    
    crr_values = []
    energy_values = []
    
    for _, row in weather.head(24).iterrows():  # First day
        crr = calculator.calculate_instantaneous_crr(
            row['temperature'],
            row['relative_humidity']
        )
        energy = calculator.calculate_energy_requirement(
            row['temperature'],
            row['relative_humidity']
        )
        crr_values.append(crr)
        energy_values.append(energy)
    
    print(f"\nFirst 24 hours performance:")
    print(f"  Average CRR:    {np.mean(crr_values):.4f} mol CO₂/(kg·h)")
    print(f"  Max CRR:        {np.max(crr_values):.4f} mol CO₂/(kg·h)")
    print(f"  Min CRR:        {np.min(crr_values):.4f} mol CO₂/(kg·h)")
    print(f"  Average energy: {np.mean(energy_values):.1f} kWh/tonne CO₂")
    
    return crr_values, energy_values


def example_location_specific_performance():
    """Example 3: Calculate performance for specific locations."""
    print("\n" + "="*70)
    print("Example 3: Location-Specific Performance Analysis")
    print("="*70)
    
    # Create geospatial performance analyzer
    geo = GeospatialPerformance()
    
    # Define locations
    locations = [
        Location("Phoenix, AZ", 33.45, -112.07),
        Location("Seattle, WA", 47.61, -122.33),
        Location("Miami, FL", 25.76, -80.19),
        Location("Anchorage, AK", 61.22, -149.90),
    ]
    
    # Climate mappings
    climate_map = {
        "Phoenix, AZ": "arid",
        "Seattle, WA": "temperate",
        "Miami, FL": "tropical",
        "Anchorage, AK": "polar",
    }
    
    # Generate weather data for each location
    loader = MERRA2DataLoader()
    
    for location in locations:
        climate = climate_map[location.name]
        weather = loader.generate_synthetic_data(n_hours=8760, location=climate)
        geo.add_location(location, weather)
    
    print(f"\nAdded {len(locations)} locations to analysis")
    
    # Define sorbent parameters
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
    
    # Compare locations
    print("\nCalculating performance for all locations...")
    comparison = geo.compare_locations(sorbent)
    
    print("\nLocation Performance Comparison:")
    print(comparison.to_string(index=False))
    
    # Plot comparison
    geo.plot_location_comparison(comparison, save_path='location_comparison.png')
    
    return comparison


def example_climate_optimization():
    """Example 4: Optimize operating conditions for climate."""
    print("\n" + "="*70)
    print("Example 4: Climate-Specific Optimization")
    print("="*70)
    
    # Generate weather data for arid climate (e.g., desert)
    loader = MERRA2DataLoader()
    weather = loader.generate_synthetic_data(n_hours=8760, location="arid")
    
    print("\nOptimizing for arid climate (desert)...")
    print("  Average temperature: {:.1f}°C".format(weather['temperature'].mean()))
    print("  Average humidity: {:.2f}".format(weather['relative_humidity'].mean()))
    
    # Define initial sorbent parameters
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
    
    # Create optimizer
    optimizer = ClimateOptimizer(sorbent, weather)
    
    # Grid search
    print("\nPerforming grid search...")
    cycle_times = np.linspace(4, 10, 7)
    regen_temps = np.linspace(85, 115, 7)
    
    grid_results = optimizer.grid_search(cycle_times, regen_temps)
    
    # Find optimal point
    optimal = grid_results.loc[grid_results['avg_crr'].idxmax()]
    
    print("\nGrid Search Results:")
    print(f"  Optimal cycle time:       {optimal['cycle_time']:.1f} hours")
    print(f"  Optimal regen temp:       {optimal['regen_temp']:.1f}°C")
    print(f"  Maximum CRR:              {optimal['avg_crr']:.4f} mol CO₂/(kg·h)")
    print(f"  Average energy:           {optimal['avg_energy']:.1f} kWh/tonne CO₂")
    
    # Try PSO optimization
    print("\nPerforming PSO optimization...")
    try:
        pso_results = optimizer.optimize_pso(
            bounds=[(4.0, 10.0), (85.0, 115.0)],
            swarmsize=20,
            maxiter=50
        )
        
        print("\nPSO Optimization Results:")
        print(f"  Optimal cycle time:       {pso_results['optimal_cycle_time']:.2f} hours")
        print(f"  Optimal regen temp:       {pso_results['optimal_regen_temp']:.1f}°C")
        print(f"  Maximum CRR:              {pso_results['max_crr']:.4f} mol CO₂/(kg·h)")
    except Exception as e:
        print(f"\nPSO optimization skipped: {e}")
    
    return grid_results


def example_hourly_performance_profile():
    """Example 5: Hourly performance profile (diurnal pattern)."""
    print("\n" + "="*70)
    print("Example 5: Hourly Performance Profile")
    print("="*70)
    
    # Generate weather data
    loader = MERRA2DataLoader()
    weather = loader.generate_synthetic_data(n_hours=8760, location="temperate")
    
    # Calculate hourly averages
    hourly = loader.get_hourly_averages()
    
    print("\nDiurnal (hourly) climate patterns:")
    print(hourly.head(12).to_string(index=False))
    
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
    
    calculator = PerformanceCalculator(sorbent)
    
    # Calculate hourly CRR
    hourly['crr'] = hourly.apply(
        lambda row: calculator.calculate_instantaneous_crr(
            row['temperature'], row['relative_humidity']
        ),
        axis=1
    )
    
    hourly['energy'] = hourly.apply(
        lambda row: calculator.calculate_energy_requirement(
            row['temperature'], row['relative_humidity']
        ),
        axis=1
    )
    
    print("\nHourly performance profile:")
    print(f"{'Hour':>5s} {'Temp (°C)':>10s} {'RH':>8s} {'CRR':>12s} {'Energy':>15s}")
    print("-" * 55)
    
    for _, row in hourly.iterrows():
        print(f"{int(row['hour']):>5d} {row['temperature']:>10.1f} "
              f"{row['relative_humidity']:>8.2f} {row['crr']:>12.4f} "
              f"{row['energy']:>15.1f}")
    
    # Plot hourly profile
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Temperature and humidity
    ax1_twin = ax1.twinx()
    ax1.plot(hourly['hour'], hourly['temperature'], 'r-', linewidth=2, label='Temperature')
    ax1_twin.plot(hourly['hour'], hourly['relative_humidity'], 'b-', linewidth=2, label='Humidity')
    
    ax1.set_ylabel('Temperature (°C)', color='r', fontsize=12)
    ax1_twin.set_ylabel('Relative Humidity', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1_twin.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Diurnal Climate Pattern', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # CRR and energy
    ax2_twin = ax2.twinx()
    ax2.plot(hourly['hour'], hourly['crr'], 'g-', linewidth=2, label='CRR')
    ax2_twin.plot(hourly['hour'], hourly['energy'], 'orange', linewidth=2, label='Energy')
    
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('CRR (mol CO₂/kg·h)', color='g', fontsize=12)
    ax2_twin.set_ylabel('Energy (kWh/tonne)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.set_title('Performance Profile', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(range(0, 25, 3))
    
    plt.tight_layout()
    plt.savefig('hourly_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'hourly_performance.png'")
    
    return hourly


def example_seasonal_analysis():
    """Example 6: Seasonal performance analysis."""
    print("\n" + "="*70)
    print("Example 6: Seasonal Performance Analysis")
    print("="*70)
    
    # Generate full year of data
    loader = MERRA2DataLoader()
    weather = loader.generate_synthetic_data(n_hours=8760, location="temperate")
    
    # Calculate monthly averages
    monthly = loader.get_monthly_averages()
    
    print("\nMonthly climate patterns:")
    print(monthly.to_string(index=False))
    
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
    
    calculator = PerformanceCalculator(sorbent)
    
    # Calculate monthly performance
    monthly['crr'] = monthly.apply(
        lambda row: calculator.calculate_instantaneous_crr(
            row['temperature'], row['relative_humidity']
        ),
        axis=1
    )
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = [month_names[i-1] for i in monthly['month']]
    
    print("\nMonthly performance:")
    print(f"{'Month':>5s} {'CRR':>12s} {'Capacity Factor':>18s}")
    print("-" * 40)
    
    max_crr = monthly['crr'].max()
    for _, row in monthly.iterrows():
        capacity_factor = (row['crr'] / max_crr) * 100
        print(f"{row['month_name']:>5s} {row['crr']:>12.4f} {capacity_factor:>17.1f}%")
    
    # Annual statistics
    annual_avg_crr = monthly['crr'].mean()
    print(f"\nAnnual average CRR: {annual_avg_crr:.4f} mol CO₂/(kg·h)")
    print(f"Best month: {monthly.loc[monthly['crr'].idxmax(), 'month_name']} "
          f"(CRR = {max_crr:.4f})")
    print(f"Worst month: {monthly.loc[monthly['crr'].idxmin(), 'month_name']} "
          f"(CRR = {monthly['crr'].min():.4f})")
    
    return monthly


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MODULE 4: GEOSPATIAL PERFORMANCE & SITING")
    print("Examples and Demonstrations")
    print("="*70 + "\n")
    
    # Run examples
    weather_datasets = example_weather_data_generation()
    crr_values, energy_values = example_crr_calculation()
    comparison = example_location_specific_performance()
    grid_results = example_climate_optimization()
    hourly_profile = example_hourly_performance_profile()
    monthly_perf = example_seasonal_analysis()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("Generated plots: location_comparison.png, hourly_performance.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
