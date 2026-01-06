import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(fitness_history, output_path='huawei_convergence.png'):
    """
    Generates an academic-style convergence plot for the optimization process.
    
    Args:
        fitness_history (list or np.array): List of fitness values per generation.
        output_path (str): Path to save the generated image.
    """
    generations = np.arange(len(fitness_history))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(generations, fitness_history, marker='o', linestyle='-', color='#d62728', linewidth=2, label='Best Fitness')
    
    max_fitness = np.max(fitness_history)
    plt.axhline(y=max_fitness, color='gray', linestyle='--', alpha=0.7, label=f'Peak Efficiency: {max_fitness:.4f}')
    
    plt.title('Evolutionary Algorithm Convergence: Solar Efficiency Optimization', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (Efficiency Metric)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.annotate(f'Start: {fitness_history[0]:.4f}', 
                 xy=(0, fitness_history[0]), 
                 xytext=(1, fitness_history[0] + (max_fitness - fitness_history[0])*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
