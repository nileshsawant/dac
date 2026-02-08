"""
Module 4: Geospatial Performance & Siting
==========================================

This module integrates real-world hourly weather data (MERRA-2) with the 
performance model to calculate location-specific Carbon Removal Rate (CRR).

Key Features:
- NASA MERRA-2 hourly weather data integration
- Dynamic cycle duration adjustment based on ambient conditions
- Particle Swarm Optimization (PSO) and Genetic Algorithms for optimization
- Carbon Removal Rate (CRR) calculation
- Multi-location performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy import optimize
from pyswarm import pso
import seaborn as sns


@dataclass
class WeatherData:
    """Container for weather data at a specific time."""
    timestamp: datetime
    temperature: float  # °C
    relative_humidity: float  # 0-1
    pressure: float  # mbar
    wind_speed: Optional[float] = None  # m/s
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'temperature': self.temperature,
            'relative_humidity': self.relative_humidity,
            'pressure': self.pressure,
            'wind_speed': self.wind_speed
        }


@dataclass
class SorbentParameters:
    """Container for sorbent performance parameters."""
    co2_capacity_ref: float  # Reference CO2 capacity at 25°C, 50% RH (mol/kg)
    working_capacity: float  # Working capacity (mol/kg)
    h2o_penalty_factor: float  # H2O co-adsorption penalty (fractional)
    temp_coefficient: float  # Temperature sensitivity (1/°C)
    humidity_coefficient: float  # Humidity sensitivity
    regeneration_temp: float  # Required regeneration temperature (°C)
    regeneration_vacuum: float  # Required vacuum level (mbar)
    cycle_time_base: float  # Base cycle time (hours)
    
    def adjust_capacity(self, temperature: float, humidity: float) -> float:
        """
        Adjust CO2 capacity based on ambient conditions.
        
        Parameters:
        -----------
        temperature : float
            Temperature (°C)
        humidity : float
            Relative humidity (0-1)
            
        Returns:
        --------
        float
            Adjusted CO2 capacity (mol/kg)
        """
        # Temperature effect (capacity decreases with temperature)
        temp_factor = 1.0 - self.temp_coefficient * (temperature - 25.0)
        
        # Humidity penalty
        humidity_penalty = self.h2o_penalty_factor * humidity * self.humidity_coefficient
        
        adjusted_capacity = self.co2_capacity_ref * temp_factor * (1.0 - humidity_penalty)
        
        return max(adjusted_capacity, 0.1)  # Ensure positive capacity


@dataclass
class Location:
    """Geographic location for DAC siting."""
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float] = None  # meters


class MERRA2DataLoader:
    """
    Load and process NASA MERRA-2 weather data.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize MERRA-2 data loader.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to MERRA-2 netCDF file or CSV
        """
        self.data_path = data_path
        self.weather_data = None
        
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load weather data from CSV file.
        
        Expected columns: timestamp, temperature, relative_humidity, pressure
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Weather data
        """
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate required columns
        required = ['timestamp', 'temperature', 'relative_humidity', 'pressure']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.weather_data = df
        return df
    
    def load_from_netcdf(self, filepath: str, 
                        lat: float, lon: float) -> pd.DataFrame:
        """
        Load weather data from MERRA-2 netCDF file for specific location.
        
        Parameters:
        -----------
        filepath : str
            Path to MERRA-2 netCDF file
        lat : float
            Latitude
        lon : float
            Longitude
            
        Returns:
        --------
        pd.DataFrame
            Weather data
        """
        dataset = nc.Dataset(filepath, 'r')
        
        # Find nearest grid point
        lats = dataset.variables['lat'][:]
        lons = dataset.variables['lon'][:]
        
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        # Extract data
        times = dataset.variables['time'][:]
        temps = dataset.variables['T2M'][:, lat_idx, lon_idx] - 273.15  # K to °C
        rh = dataset.variables['RH2M'][:, lat_idx, lon_idx] / 100.0  # % to fraction
        pressure = dataset.variables['PS'][:, lat_idx, lon_idx] / 100.0  # Pa to mbar
        
        # Create DataFrame
        base_time = datetime(1980, 1, 1)  # MERRA-2 reference time
        timestamps = [base_time + timedelta(minutes=int(t)) for t in times]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temps,
            'relative_humidity': rh,
            'pressure': pressure
        })
        
        dataset.close()
        self.weather_data = df
        return df
    
    def generate_synthetic_data(self, 
                               n_hours: int = 8760,
                               location: str = "temperate") -> pd.DataFrame:
        """
        Generate synthetic hourly weather data for testing.
        
        Parameters:
        -----------
        n_hours : int
            Number of hours (default: 8760 = 1 year)
        location : str
            Climate type: "temperate", "tropical", "arid", "polar"
            
        Returns:
        --------
        pd.DataFrame
            Synthetic weather data
        """
        timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) 
                     for i in range(n_hours)]
        
        # Climate profiles
        profiles = {
            "temperate": {"temp_mean": 15, "temp_amp": 15, "rh_mean": 0.60},
            "tropical": {"temp_mean": 27, "temp_amp": 5, "rh_mean": 0.80},
            "arid": {"temp_mean": 25, "temp_amp": 20, "rh_mean": 0.25},
            "polar": {"temp_mean": -5, "temp_amp": 10, "rh_mean": 0.70},
        }
        
        profile = profiles.get(location, profiles["temperate"])
        
        # Generate data with seasonal and diurnal variations
        hours = np.arange(n_hours)
        
        # Temperature (seasonal + diurnal)
        seasonal = profile["temp_amp"] * np.sin(2 * np.pi * hours / 8760)
        diurnal = 5 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 2, n_hours)
        temperature = profile["temp_mean"] + seasonal + diurnal + noise
        
        # Relative humidity (inverse correlation with temperature)
        rh_base = profile["rh_mean"]
        rh_variation = -0.1 * (temperature - profile["temp_mean"]) / profile["temp_amp"]
        rh_noise = np.random.normal(0, 0.05, n_hours)
        relative_humidity = np.clip(rh_base + rh_variation + rh_noise, 0.1, 0.95)
        
        # Pressure (slight seasonal variation)
        pressure_base = 1013
        pressure_variation = 20 * np.sin(2 * np.pi * hours / 8760)
        pressure_noise = np.random.normal(0, 5, n_hours)
        pressure = pressure_base + pressure_variation + pressure_noise
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'relative_humidity': relative_humidity,
            'pressure': pressure
        })
        
        self.weather_data = df
        return df
    
    def get_hourly_averages(self) -> pd.DataFrame:
        """
        Calculate hourly averages (diurnal pattern).
        
        Returns:
        --------
        pd.DataFrame
            Average conditions by hour of day
        """
        if self.weather_data is None:
            raise ValueError("No weather data loaded")
        
        df = self.weather_data.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_avg = df.groupby('hour').agg({
            'temperature': 'mean',
            'relative_humidity': 'mean',
            'pressure': 'mean'
        }).reset_index()
        
        return hourly_avg
    
    def get_monthly_averages(self) -> pd.DataFrame:
        """
        Calculate monthly averages.
        
        Returns:
        --------
        pd.DataFrame
            Average conditions by month
        """
        if self.weather_data is None:
            raise ValueError("No weather data loaded")
        
        df = self.weather_data.copy()
        df['month'] = df['timestamp'].dt.month
        
        monthly_avg = df.groupby('month').agg({
            'temperature': 'mean',
            'relative_humidity': 'mean',
            'pressure': 'mean'
        }).reset_index()
        
        return monthly_avg


class PerformanceCalculator:
    """
    Calculate DAC performance metrics from weather data and sorbent parameters.
    """
    
    def __init__(self, 
                 sorbent_params: SorbentParameters,
                 degradation_rate: float = 5e-6):
        """
        Initialize performance calculator.
        
        Parameters:
        -----------
        sorbent_params : SorbentParameters
            Sorbent performance parameters
        degradation_rate : float
            Degradation rate constant (cycle^-1)
        """
        self.sorbent = sorbent_params
        self.degradation_rate = degradation_rate
        
    def calculate_cycle_time(self, 
                            temperature: float,
                            humidity: float) -> float:
        """
        Calculate optimal cycle time based on ambient conditions.
        
        Parameters:
        -----------
        temperature : float
            Ambient temperature (°C)
        humidity : float
            Relative humidity (0-1)
            
        Returns:
        --------
        float
            Cycle time (hours)
        """
        # Base cycle time adjusted for conditions
        # Higher temperature -> shorter adsorption time
        # Higher humidity -> longer purging/regeneration time
        
        temp_factor = 1.0 / (1.0 + 0.02 * max(temperature - 25, 0))
        humidity_factor = 1.0 + 0.5 * humidity
        
        cycle_time = self.sorbent.cycle_time_base * temp_factor * humidity_factor
        
        return cycle_time
    
    def calculate_instantaneous_crr(self,
                                   temperature: float,
                                   humidity: float,
                                   cycle_number: int = 0) -> float:
        """
        Calculate instantaneous Carbon Removal Rate (CRR).
        
        CRR = capacity × (cycles/time) × (1 - degradation)
        
        Parameters:
        -----------
        temperature : float
            Ambient temperature (°C)
        humidity : float
            Relative humidity (0-1)
        cycle_number : int
            Current cycle number (for degradation calculation)
            
        Returns:
        --------
        float
            CRR in mol CO2 / (kg sorbent · hour)
        """
        # Get adjusted capacity
        capacity = self.sorbent.adjust_capacity(temperature, humidity)
        
        # Get cycle time
        cycle_time = self.calculate_cycle_time(temperature, humidity)
        
        # Account for degradation
        degradation_factor = np.exp(-self.degradation_rate * cycle_number)
        
        # CRR = capacity per cycle / cycle time
        crr = (capacity * degradation_factor) / cycle_time
        
        return crr
    
    def calculate_energy_requirement(self,
                                    temperature: float,
                                    humidity: float) -> float:
        """
        Calculate specific energy requirement (kWh/tonne CO2).
        
        Parameters:
        -----------
        temperature : float
            Ambient temperature (°C)
        humidity : float
            Relative humidity (0-1)
            
        Returns:
        --------
        float
            Specific energy (kWh/tonne CO2)
        """
        # Simplified energy model
        # Components: heating, vacuum, cooling
        
        # Heating energy (proportional to temperature lift)
        temp_lift = self.sorbent.regeneration_temp - temperature
        heating_energy = 2.5 * temp_lift  # kWh/tonne per °C
        
        # Vacuum energy (proportional to pressure ratio)
        vacuum_energy = 50 * np.log(1013 / self.sorbent.regeneration_vacuum)
        
        # Water removal penalty
        water_penalty = 100 * humidity  # kWh/tonne
        
        total_energy = heating_energy + vacuum_energy + water_penalty
        
        return total_energy


class ClimateOptimizer:
    """
    Optimize DAC operating conditions for different climate zones.
    """
    
    def __init__(self, 
                 sorbent_params: SorbentParameters,
                 weather_data: pd.DataFrame):
        """
        Initialize climate optimizer.
        
        Parameters:
        -----------
        sorbent_params : SorbentParameters
            Sorbent performance parameters
        weather_data : pd.DataFrame
            Hourly weather data
        """
        self.sorbent = sorbent_params
        self.weather_data = weather_data
        self.calculator = PerformanceCalculator(sorbent_params)
        
    def objective_function(self, 
                          params: np.ndarray,
                          weather_subset: pd.DataFrame) -> float:
        """
        Objective function for optimization (minimize negative CRR).
        
        Parameters:
        -----------
        params : np.ndarray
            [cycle_time_base, regeneration_temp]
        weather_subset : pd.DataFrame
            Weather data subset
            
        Returns:
        --------
        float
            Negative average CRR (for minimization)
        """
        # Update sorbent parameters
        cycle_time_base, regen_temp = params
        
        self.sorbent.cycle_time_base = cycle_time_base
        self.sorbent.regeneration_temp = regen_temp
        
        # Calculate average CRR
        crr_values = []
        for _, row in weather_subset.iterrows():
            crr = self.calculator.calculate_instantaneous_crr(
                row['temperature'],
                row['relative_humidity']
            )
            crr_values.append(crr)
        
        avg_crr = np.mean(crr_values)
        
        return -avg_crr  # Negative for minimization
    
    def optimize_pso(self,
                    bounds: Optional[List[Tuple[float, float]]] = None,
                    swarmsize: int = 30,
                    maxiter: int = 100) -> Dict:
        """
        Optimize using Particle Swarm Optimization.
        
        Parameters:
        -----------
        bounds : List[Tuple[float, float]], optional
            Parameter bounds [(cycle_min, cycle_max), (temp_min, temp_max)]
        swarmsize : int
            Number of particles
        maxiter : int
            Maximum iterations
            
        Returns:
        --------
        Dict
            Optimization results
        """
        if bounds is None:
            bounds = [(2.0, 12.0), (80.0, 120.0)]  # cycle time, regen temp
        
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        
        # Run PSO
        xopt, fopt = pso(
            lambda x: self.objective_function(x, self.weather_data),
            lb, ub,
            swarmsize=swarmsize,
            maxiter=maxiter
        )
        
        return {
            'optimal_cycle_time': xopt[0],
            'optimal_regen_temp': xopt[1],
            'max_crr': -fopt,
            'method': 'PSO'
        }
    
    def grid_search(self,
                   cycle_times: np.ndarray,
                   regen_temps: np.ndarray) -> pd.DataFrame:
        """
        Perform grid search over parameter space.
        
        Parameters:
        -----------
        cycle_times : np.ndarray
            Array of cycle times to test
        regen_temps : np.ndarray
            Array of regeneration temperatures to test
            
        Returns:
        --------
        pd.DataFrame
            Grid search results
        """
        results = []
        
        for ct in cycle_times:
            for rt in regen_temps:
                self.sorbent.cycle_time_base = ct
                self.sorbent.regeneration_temp = rt
                
                # Calculate average CRR
                crr_values = []
                energy_values = []
                
                for _, row in self.weather_data.iterrows():
                    crr = self.calculator.calculate_instantaneous_crr(
                        row['temperature'],
                        row['relative_humidity']
                    )
                    energy = self.calculator.calculate_energy_requirement(
                        row['temperature'],
                        row['relative_humidity']
                    )
                    crr_values.append(crr)
                    energy_values.append(energy)
                
                results.append({
                    'cycle_time': ct,
                    'regen_temp': rt,
                    'avg_crr': np.mean(crr_values),
                    'avg_energy': np.mean(energy_values)
                })
        
        return pd.DataFrame(results)


class GeospatialPerformance:
    """
    Main class for geospatial performance analysis and siting.
    """
    
    def __init__(self):
        """Initialize geospatial performance analyzer."""
        self.locations = {}
        self.performance_data = {}
        
    def add_location(self, 
                    location: Location,
                    weather_data: pd.DataFrame):
        """
        Add a location with its weather data.
        
        Parameters:
        -----------
        location : Location
            Location information
        weather_data : pd.DataFrame
            Hourly weather data for location
        """
        self.locations[location.name] = location
        self.performance_data[location.name] = weather_data
        
    def calculate_crr(self,
                     location_name: str,
                     sorbent_params: SorbentParameters,
                     n_years: int = 1) -> pd.DataFrame:
        """
        Calculate CRR time series for a location.
        
        Parameters:
        -----------
        location_name : str
            Name of location
        sorbent_params : SorbentParameters
            Sorbent parameters
        n_years : int
            Number of years to simulate
            
        Returns:
        --------
        pd.DataFrame
            Time series of CRR and performance metrics
        """
        if location_name not in self.performance_data:
            raise ValueError(f"Location {location_name} not found")
        
        weather = self.performance_data[location_name]
        calculator = PerformanceCalculator(sorbent_params)
        
        results = []
        cycle_number = 0
        
        for idx, row in weather.iterrows():
            crr = calculator.calculate_instantaneous_crr(
                row['temperature'],
                row['relative_humidity'],
                cycle_number
            )
            
            energy = calculator.calculate_energy_requirement(
                row['temperature'],
                row['relative_humidity']
            )
            
            cycle_time = calculator.calculate_cycle_time(
                row['temperature'],
                row['relative_humidity']
            )
            
            results.append({
                'timestamp': row['timestamp'],
                'temperature': row['temperature'],
                'humidity': row['relative_humidity'],
                'crr': crr,
                'energy': energy,
                'cycle_time': cycle_time
            })
            
            cycle_number += 1 / cycle_time
        
        return pd.DataFrame(results)
    
    def compare_locations(self,
                         sorbent_params: SorbentParameters) -> pd.DataFrame:
        """
        Compare performance across all locations.
        
        Parameters:
        -----------
        sorbent_params : SorbentParameters
            Sorbent parameters
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        comparison = []
        
        for loc_name in self.locations.keys():
            crr_data = self.calculate_crr(loc_name, sorbent_params)
            
            comparison.append({
                'location': loc_name,
                'avg_crr': crr_data['crr'].mean(),
                'max_crr': crr_data['crr'].max(),
                'min_crr': crr_data['crr'].min(),
                'avg_energy': crr_data['energy'].mean(),
                'capacity_factor': (crr_data['crr'].mean() / crr_data['crr'].max()) * 100
            })
        
        return pd.DataFrame(comparison).sort_values('avg_crr', ascending=False)
    
    def plot_location_comparison(self,
                                comparison: pd.DataFrame,
                                save_path: Optional[str] = None):
        """
        Plot comparison of locations.
        
        Parameters:
        -----------
        comparison : pd.DataFrame
            Comparison results
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Average CRR
        ax1.barh(comparison['location'], comparison['avg_crr'])
        ax1.set_xlabel('Average CRR (mol CO₂/kg·h)', fontsize=12)
        ax1.set_title('Carbon Removal Rate by Location', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # Plot 2: Energy vs CRR
        ax2.scatter(comparison['avg_energy'], comparison['avg_crr'], 
                   s=200, alpha=0.6, c=range(len(comparison)), cmap='viridis')
        
        for idx, row in comparison.iterrows():
            ax2.annotate(row['location'], 
                        (row['avg_energy'], row['avg_crr']),
                        fontsize=9, ha='right')
        
        ax2.set_xlabel('Average Energy (kWh/tonne CO₂)', fontsize=12)
        ax2.set_ylabel('Average CRR (mol CO₂/kg·h)', fontsize=12)
        ax2.set_title('Energy Efficiency Trade-off', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
