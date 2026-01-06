import sys
import os
import time
import numpy as np
import mindspore as ms
from mindspore import Tensor
from physics.SolarOptimizationManager import SolarOptimizationManager
import benchmarks.config_bench as config
from benchmarks.visualizer import plot_convergence

def run_benchmark():
    print("===============================================================")
    print("    BENCHMARK TESTS - SOLAR OPTIMIZATION MANAGER v1.0")
    print("===============================================================")
    
    print("\n--- Initializing System ---")
    
    data_path = os.path.join("data", "materials.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    manager = SolarOptimizationManager(csv_path=data_path)
    
    wavelengths = np.linspace(
        config.SIMULATION_CONFIG['wavelength_range'][0],
        config.SIMULATION_CONFIG['wavelength_range'][1],
        config.SIMULATION_CONFIG['wavelength_steps']
    )
    wavelength_tensor = Tensor(wavelengths, ms.float32)
    
    params = config.SIMULATION_CONFIG.copy()
    params['wavelength'] = wavelength_tensor
    
    print("System initialized.")
    print(f"Target Materials: {params['materials']}")
    print(f"Population: {params['pop_size']}, Generations: {params['iterations']}")

    print("\n--- Executing Study ---")
    
    start_time = time.time()
    
    fitness_history, best_radii = manager.run_study(params)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Study complete in {execution_time:.4f} seconds.")
    print(f"Best Radii Found: {best_radii}")

    print("\n--- Auditing & Calculating Metrics ---")
    
    total_candidates = params['pop_size'] * params['iterations']
    
    # Test 1: Computational Performance
    throughput = total_candidates / execution_time if execution_time > 0 else 0
    perf_status = "PASS" if execution_time < config.KPIS['max_execution_time_sec'] else "FAIL"
    
    # Test 2: Economic Impact
    lab_cost_total = total_candidates * config.LAB_COST_PER_EXPERIMENT_USD
    lab_time_total_hours = total_candidates * config.LAB_TIME_PER_EXPERIMENT_HOURS
    lab_time_total_years = lab_time_total_hours / (24 * 365)
    acceleration_factor = (lab_time_total_hours * 3600) / execution_time 
    
    # Test 3: Convergence & Improvement
    if isinstance(fitness_history, Tensor):
        fitness_history = fitness_history.asnumpy().tolist()
    elif isinstance(fitness_history, np.ndarray):
        fitness_history = fitness_history.tolist()
        
    initial_fitness = fitness_history[0]
    final_fitness = fitness_history[-1]
    improvement_pct = ((final_fitness - initial_fitness) / abs(initial_fitness)) * 100
    
    plot_convergence(fitness_history, output_path='benchmarks/output/convergence.png')
    
    # Test 4: Physics Precision 
    test_material = config.PHYSICS_GROUND_TRUTH['material']
    test_radius = config.PHYSICS_GROUND_TRUTH['test_radius_nm']
    expected_bg = config.PHYSICS_GROUND_TRUTH['expected_bandgap_ev']
    
    engine = manager.get_engine(test_material)
    
    dummy_wavelengths = Tensor([500.0], ms.float32) 
    test_radius_tensor = Tensor(test_radius, ms.float32)
    test_temp_tensor = Tensor(config.SIMULATION_CONFIG['temp'], ms.float32)
    
    # Returns (absorption_spectrum, bandgap)
    _, calculated_bg_tensor = engine(test_temp_tensor, test_radius_tensor, dummy_wavelengths)
    calculated_bg = float(calculated_bg_tensor.asnumpy())
    
    error_pct = abs((calculated_bg - expected_bg) / expected_bg) * 100
    physics_status = "PASS" if error_pct < config.KPIS['max_physics_error_percent'] else "FAIL"
    
    # Test 5: Sustainability
    waste_saved_kg = (total_candidates * config.CHEMICAL_WASTE_PER_EXPERIMENT_G) / 1000.0
    
    print("\n" + "-"*30)
    print("   BENCHMARK REPORT RESULTS")
    print("-"*30)
    
    print(f"\n[1] PERFORMANCE (MindSpore)")
    print(f"    - Execution Time: {execution_time:.4f} s (Target: <{config.KPIS['max_execution_time_sec']}s) [{perf_status}]")
    print(f"    - Throughput:     {throughput:.2f} designs/sec")
    
    print(f"\n[2] ECONOMIC VIABILITY")
    print(f"    - Virtual Cost:   $0.00 (approx)")
    print(f"    - Lab Cost Saved: ${lab_cost_total:,.2f} USD")
    print(f"    - Time Saved:     {lab_time_total_years:.2f} years")
    print(f"    - Speedup Factor: {acceleration_factor:,.0f}x")
    
    print(f"\n[3] ARTIFICIAL INTELLIGENCE")
    print(f"    - Generations:    {params['iterations']}")
    print(f"    - Initial Fitness:    {initial_fitness:.4f}")
    print(f"    - Final Best Fitness: {final_fitness:.4f}")
    print(f"    - Improvement:    {improvement_pct:.2f}%")
    print(f"    - Convergence:    See 'huawei_convergence.png'")
    
    print(f"\n[4] PHYSICS ACCURACY")
    print(f"    - Material:       {test_material} @ {test_radius}nm")
    print(f"    - Expected Eg:    {expected_bg:.4f} eV")
    print(f"    - Calculated Eg:  {calculated_bg:.4f} eV")
    print(f"    - Error:          {error_pct:.4f}% (Target: <{config.KPIS['max_physics_error_percent']}%) [{physics_status}]")
    
    print(f"\n[5] SUSTAINABILITY")
    print(f"    - Toxic Waste Avoided: {waste_saved_kg:.2f} kg")
    
    print("\n" + "-"*60)
    if perf_status == "PASS" and physics_status == "PASS":
        print("   SUCCESS: System passed all critical benchmarks.")
    else:
        print("   WARNING: System failed one or more benchmarks.")
    print("-"*60)

if __name__ == "__main__":
    run_benchmark()
