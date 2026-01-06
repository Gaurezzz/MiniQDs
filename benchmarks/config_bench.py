"""
Configuration file for the Huawei Benchmarking Suite.
Acts as the single source of truth for business metrics, simulation parameters, and physics validation.
"""

# --- Business Constants ---
LAB_COST_PER_EXPERIMENT_USD = 150.0  # Cost of materials, equipment usage, and personnel per sample
LAB_TIME_PER_EXPERIMENT_HOURS = 72.0 # Average time to synthesize and characterize one sample (3 days)
CHEMICAL_WASTE_PER_EXPERIMENT_G = 45.0 # Grams of toxic waste generated per physical experiment

# --- Simulation Parameters (MindSpore Context) ---
SIMULATION_CONFIG = {
    'materials': ['PbS', 'CdSe'],
    'pop_size': 100,          # Number of candidates per generation
    'iterations': 50,        # Number of evolutionary generations
    'alpha': 0.1,            # Learning rate
    'mutation': 0.1,         # Mutation probability/strength
    'temp': 298.0,           # Temperature in Kelvin
    'wavelength_range': [300.0, 2500.0], # Wavelength range in nm for tensor creation
    'wavelength_steps': 100,  # Resolution of the spectrum
}

# --- Physics Ground Truth ---
# Reference values from literature for validation
PHYSICS_GROUND_TRUTH = {
    'material': 'PbS',
    'test_radius_nm': 3.0,
    'expected_bandgap_ev': 1.3192, # Approximate value for PbS at 3nm 
    'tolerance_percent': 1.0,     # Maximum allowed error percentage

}

# --- KPI Targets ---
KPIS = {
    'max_execution_time_sec': 120.0,
    'min_cost_saving_usd': 10000.0,
    'max_physics_error_percent': 5.0,
    'min_throughput_candidates_per_sec': 10.0
}
