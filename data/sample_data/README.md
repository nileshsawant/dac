# Sample Data Directory

This directory contains sample data files for testing and demonstration.

## Contents

### 1. QM Output Examples

Place your Gaussian/VASP output files here for parsing with Module 1:

- `sample_gaussian.log` - Example Gaussian output
- `sample_OUTCAR` - Example VASP output

### 2. Weather Data

Sample MERRA-2 or custom weather data:

- `sample_weather_temperate.csv` - Temperate climate (1 year hourly)
- `sample_weather_arid.csv` - Arid climate
- `sample_weather_tropical.csv` - Tropical climate

Expected CSV format:
```
timestamp,temperature,relative_humidity,pressure
2023-01-01 00:00:00,15.2,0.65,1013.2
2023-01-01 01:00:00,14.8,0.68,1012.8
...
```

### 3. Molecular Structures

For Module 2 (GNN training):

- `molecules/` - Directory for SMILES strings or SDF files
  - `co2_amine_complexes.csv` - CO2-amine binding products
  - `sorbent_structures.sdf` - 3D molecular structures

### 4. Performance Data

Experimental or simulated performance data:

- `performance_data.csv` - Measured CO2 capacity, H2O penalty, etc.

Format:
```
sorbent_id,co2_capacity,h2o_penalty,regen_temp
1,2.5,0.15,95
2,2.2,0.20,100
...
```

## Generating Sample Data

Use the provided scripts to generate synthetic data:

```python
# Generate weather data
from dac_framework import MERRA2DataLoader

loader = MERRA2DataLoader()
weather = loader.generate_synthetic_data(n_hours=8760, location="temperate")
weather.to_csv('sample_data/sample_weather_temperate.csv', index=False)
```

## Using Your Own Data

Replace sample files with your actual data:

1. **QM Outputs**: Direct outputs from Gaussian/VASP calculations
2. **Weather Data**: Download from [NASA MERRA-2](https://disc.gsfc.nasa.gov/datasets?project=MERRA-2)
3. **Molecular Data**: Export from molecular modeling software
4. **Performance Data**: Experimental measurements from your lab

## Data Format Guidelines

### Weather Data CSV
- **Required columns**: `timestamp`, `temperature`, `relative_humidity`, `pressure`
- **Optional columns**: `wind_speed`, `solar_radiation`
- **Units**: Temperature (°C), Humidity (0-1), Pressure (mbar)

### Performance Data CSV
- **Required columns**: `co2_capacity`, `h2o_penalty`, `regen_temperature`
- **Optional columns**: `regen_vacuum`, `cycle_time`, `energy_requirement`
- **Units**: Capacity (mol/kg), Penalty (fraction), Temperature (°C)

### Molecular Structures
- **SMILES format**: One column with SMILES strings
- **SDF format**: Standard SDF with 3D coordinates
- **Include**: Reactants and products for differential descriptors
