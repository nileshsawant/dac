"""
Utility Functions for DAC Framework

This module provides helper functions for data processing, visualization,
and analysis across all modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def setup_plotting_style():
    """Setup publication-quality plotting style."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for an array.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'range': np.max(data) - np.min(data),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
    }


def export_results(results: pd.DataFrame, 
                  filepath: str,
                  include_metadata: bool = True):
    """
    Export results to CSV with optional metadata.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results dataframe
    filepath : str
        Output file path
    include_metadata : bool
        Include metadata header
    """
    if include_metadata:
        with open(filepath, 'w') as f:
            f.write(f"# DAC Framework Results\n")
            f.write(f"# Generated: {datetime.now()}\n")
            f.write(f"# Shape: {results.shape}\n")
            f.write(f"# Columns: {', '.join(results.columns)}\n")
            f.write("#\n")
    
    results.to_csv(filepath, mode='a' if include_metadata else 'w', index=False)
    print(f"Results exported to {filepath}")


def create_comparison_table(data: Dict[str, pd.DataFrame],
                           metrics: List[str]) -> pd.DataFrame:
    """
    Create comparison table from multiple datasets.
    
    Parameters:
    -----------
    data : Dict[str, pd.DataFrame]
        Dictionary of dataframes with labels as keys
    metrics : List[str]
        List of metric column names to compare
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for label, df in data.items():
        row = {'Configuration': label}
        for metric in metrics:
            if metric in df.columns:
                row[metric] = df[metric].mean()
        comparison.append(row)
    
    return pd.DataFrame(comparison)


def plot_sensitivity_analysis(param_values: np.ndarray,
                             results: np.ndarray,
                             param_name: str,
                             metric_name: str,
                             save_path: Optional[str] = None):
    """
    Plot sensitivity analysis results.
    
    Parameters:
    -----------
    param_values : np.ndarray
        Array of parameter values tested
    results : np.ndarray
        Array of results for each parameter value
    param_name : str
        Name of parameter
    metric_name : str
        Name of metric
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, results, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Sensitivity Analysis: {metric_name} vs {param_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Highlight optimal point
    optimal_idx = np.argmax(results)
    ax.plot(param_values[optimal_idx], results[optimal_idx], 
           'r*', markersize=20, label='Optimal')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_lcoc_simple(crr: float,
                         energy: float,
                         sorbent_cost: float = 50.0,
                         energy_cost: float = 0.05,
                         plant_lifetime: float = 20.0) -> float:
    """
    Calculate simplified Levelized Cost of Carbon (LCOC).
    
    LCOC = (Capital + Operating) / Lifetime CO2 Captured
    
    Parameters:
    -----------
    crr : float
        Carbon Removal Rate (mol CO2/kg/h)
    energy : float
        Specific energy requirement (kWh/tonne CO2)
    sorbent_cost : float
        Sorbent cost ($/kg), default: 50
    energy_cost : float
        Energy cost ($/kWh), default: 0.05
    plant_lifetime : float
        Plant lifetime (years), default: 20
        
    Returns:
    --------
    float
        LCOC ($/tonne CO2)
    """
    # Convert CRR to annual capture (assuming 8760 hours/year)
    # mol CO2/kg/h * 8760 h/year * 44 g/mol / 1e6 g/tonne = tonne CO2/kg/year
    annual_capture_per_kg = crr * 8760 * 44 / 1e6
    
    # Lifetime capture per kg sorbent
    lifetime_capture = annual_capture_per_kg * plant_lifetime
    
    # Costs
    sorbent_cost_per_tonne = sorbent_cost / lifetime_capture
    energy_cost_per_tonne = energy * energy_cost
    
    lcoc = sorbent_cost_per_tonne + energy_cost_per_tonne
    
    return lcoc


def interpolate_weather_data(weather_df: pd.DataFrame,
                            target_resolution: str = '1H') -> pd.DataFrame:
    """
    Interpolate weather data to target resolution.
    
    Parameters:
    -----------
    weather_df : pd.DataFrame
        Weather dataframe with timestamp column
    target_resolution : str
        Target resolution (pandas frequency string)
        
    Returns:
    --------
    pd.DataFrame
        Interpolated weather data
    """
    df = weather_df.copy()
    df.set_index('timestamp', inplace=True)
    
    # Create target time range
    new_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=target_resolution
    )
    
    # Interpolate
    df_interp = df.reindex(df.index.union(new_index)).interpolate('linear')
    df_interp = df_interp.loc[new_index]
    
    df_interp.reset_index(inplace=True)
    df_interp.rename(columns={'index': 'timestamp'}, inplace=True)
    
    return df_interp


def analyze_performance_variability(results: pd.DataFrame,
                                   metric: str = 'crr') -> Dict:
    """
    Analyze performance variability over time.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Performance results with timestamp column
    metric : str
        Metric to analyze
        
    Returns:
    --------
    Dict
        Variability analysis
    """
    data = results[metric].values
    
    analysis = {
        'mean': np.mean(data),
        'std': np.std(data),
        'cv': np.std(data) / np.mean(data),  # Coefficient of variation
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'stability_score': 1.0 - (np.std(data) / np.mean(data)),  # 1 = perfectly stable
    }
    
    return analysis


def create_summary_report(results: Dict[str, pd.DataFrame],
                         output_file: str = 'summary_report.txt'):
    """
    Create text summary report from results.
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Dictionary of results dataframes
    output_file : str
        Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DAC FRAMEWORK ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        for name, df in results.items():
            f.write("-"*80 + "\n")
            f.write(f"{name}\n")
            f.write("-"*80 + "\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {', '.join(df.columns)}\n\n")
            
            # Numeric summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                f.write("Numeric Summary:\n")
                f.write(df[numeric_cols].describe().to_string())
                f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to {output_file}")


def validate_input_data(weather_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate weather data format and content.
    
    Parameters:
    -----------
    weather_df : pd.DataFrame
        Weather dataframe to validate
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list of error messages)
    """
    errors = []
    
    # Check required columns
    required_cols = ['timestamp', 'temperature', 'relative_humidity', 'pressure']
    missing_cols = [col for col in required_cols if col not in weather_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if 'timestamp' in weather_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(weather_df['timestamp']):
            errors.append("'timestamp' column must be datetime type")
    
    # Check value ranges
    if 'temperature' in weather_df.columns:
        temp = weather_df['temperature']
        if temp.min() < -50 or temp.max() > 60:
            errors.append(f"Temperature out of range: [{temp.min()}, {temp.max()}]")
    
    if 'relative_humidity' in weather_df.columns:
        rh = weather_df['relative_humidity']
        if rh.min() < 0 or rh.max() > 1:
            errors.append(f"Relative humidity out of range [0,1]: [{rh.min()}, {rh.max()}]")
    
    # Check for missing values
    if weather_df.isnull().any().any():
        null_cols = weather_df.columns[weather_df.isnull().any()].tolist()
        errors.append(f"Columns with missing values: {null_cols}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# Convenience function for quick analysis
def quick_analysis(weather_file: str,
                  sorbent_params: Dict,
                  n_cycles: int = 1000) -> Dict:
    """
    Perform quick integrated analysis.
    
    Parameters:
    -----------
    weather_file : str
        Path to weather CSV file
    sorbent_params : Dict
        Sorbent parameters
    n_cycles : int
        Number of degradation cycles
        
    Returns:
    --------
    Dict
        Analysis results
    """
    from dac_framework import (
        MERRA2DataLoader,
        SorbentParameters,
        PerformanceCalculator,
        MarkovDegradation
    )
    
    # Load weather
    loader = MERRA2DataLoader()
    weather = loader.load_from_csv(weather_file)
    
    # Create sorbent
    sorbent = SorbentParameters(**sorbent_params)
    
    # Calculate performance
    calculator = PerformanceCalculator(sorbent)
    crr_avg = np.mean([
        calculator.calculate_instantaneous_crr(row['temperature'], row['relative_humidity'])
        for _, row in weather.head(100).iterrows()
    ])
    
    # Simulate degradation
    markov = MarkovDegradation()
    deg_results = markov.simulate_cycles(n_cycles=n_cycles, record_interval=100)
    
    return {
        'avg_crr': crr_avg,
        'active_fraction': deg_results.iloc[-1]['active'],
        'weather_hours': len(weather),
        'cycles_simulated': n_cycles
    }


if __name__ == "__main__":
    # Example usage
    setup_plotting_style()
    
    # Test statistics calculation
    data = np.random.normal(100, 15, 1000)
    stats = calculate_statistics(data)
    print("Statistics:", stats)
