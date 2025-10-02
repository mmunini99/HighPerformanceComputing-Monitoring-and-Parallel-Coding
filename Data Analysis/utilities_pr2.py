import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_mpi_strong_scaling(data_frame, plot_title="MPI Strong Scaling Analysis", x_label="MPI Ranks"):
    """
    Analyze and visualize strong scaling performance with Amdahl's Law.
    
    Args:
        data_frame: DataFrame containing 'cores' and 'time' columns
        plot_title: Title for the speedup plot
        x_label: Label for x-axis
    """
    # Compute speedup and parallel efficiency
    baseline_time = data_frame['time'].iloc[0]
    data_frame['speedup'] = baseline_time / data_frame['time']
    data_frame['efficiency'] = data_frame['speedup'] / data_frame['cores']
    
    # Amdahl's Law model: S(n) = 1 / (s + (1-s)/n)
    def amdahl_model(n, serial_fraction):
        return 1 / (serial_fraction + (1 - serial_fraction) / n)
    
    # Prepare data for curve fitting
    num_procs = data_frame['cores'].values
    measured_speedup = data_frame['speedup'].values
    
    # Fit Amdahl's model to observed data
    fitted_params, _ = curve_fit(amdahl_model, num_procs, measured_speedup, bounds=(0, 1))
    serial_frac = fitted_params[0]
    predicted_speedup = amdahl_model(num_procs, serial_frac)
    
    # Speedup visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(num_procs, measured_speedup, marker='s', markersize=8, linestyle='', 
             color="#E91E63", label='Measured Speedup', linewidth=2)
    ax1.plot(num_procs, num_procs, linestyle='-.', color='#4CAF50', 
             label='Perfect Speedup', linewidth=2)
    ax1.plot(num_procs, predicted_speedup, linestyle='-', color='#FF9800', 
             label=f"Amdahl Model (s={serial_frac:.3f})", linewidth=2)
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Efficiency visualization
    mean_eff = data_frame['efficiency'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_frame['cores'], data_frame['efficiency'], marker='s', markersize=8,
             color="#E91E63", linestyle='None', label='Measured Efficiency', linewidth=2)
    ax2.axhline(y=1.0, color='#4CAF50', linestyle='-.', linewidth=2, label='Ideal Efficiency')
    ax2.axhline(y=mean_eff, color='#9E9E9E', linestyle='-', linewidth=2,
                label=f'Mean Efficiency ({mean_eff:.3f})')
    ax2.set_xlabel('MPI Rank Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.set_title('MPI Strong Scaling: Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
    return


def analyze_openmp_strong_scaling(data_frame, plot_title="OpenMP Strong Scaling Analysis", x_label="Thread Count"):
    """
    Analyze and visualize OpenMP strong scaling with Amdahl's Law.
    
    Args:
        data_frame: DataFrame containing 'threads' and 'time' columns
        plot_title: Title for the speedup plot
        x_label: Label for x-axis
    """
    # Compute speedup and parallel efficiency
    baseline_time = data_frame['time'].iloc[0]
    data_frame['speedup'] = baseline_time / data_frame['time']
    data_frame['efficiency'] = data_frame['speedup'] / data_frame['threads']
    
    # Amdahl's Law model
    def amdahl_model(n, serial_fraction):
        return 1 / (serial_fraction + (1 - serial_fraction) / n)
    
    # Prepare data for curve fitting
    num_threads = data_frame['threads'].values
    measured_speedup = data_frame['speedup'].values
    
    # Fit Amdahl's model to observed data
    fitted_params, _ = curve_fit(amdahl_model, num_threads, measured_speedup, bounds=(0, 1))
    serial_frac = fitted_params[0]
    predicted_speedup = amdahl_model(num_threads, serial_frac)
    
    # Speedup visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(num_threads, measured_speedup, marker='D', markersize=8, linestyle='',
             color="#2196F3", label='Measured Speedup', linewidth=2)
    ax1.plot(num_threads, num_threads, linestyle='-.', color='#4CAF50',
             label='Perfect Speedup', linewidth=2)
    ax1.plot(num_threads, predicted_speedup, linestyle='-', color='#FF9800',
             label=f"Amdahl Model (s={serial_frac:.3f})", linewidth=2)
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Efficiency visualization
    mean_eff = data_frame['efficiency'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_frame['threads'], data_frame['efficiency'], marker='D', markersize=8,
             color="#2196F3", linestyle='None', label='Measured Efficiency', linewidth=2)
    ax2.axhline(y=1.0, color='#4CAF50', linestyle='-.', linewidth=2, label='Ideal Efficiency')
    ax2.axhline(y=mean_eff, color='#9E9E9E', linestyle='-', linewidth=2,
                label=f'Mean Efficiency ({mean_eff:.3f})')
    ax2.set_xlabel('OpenMP Thread Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.set_title('OpenMP Strong Scaling: Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
    return


def analyze_mpi_weak_scaling(data_frame, plot_title="MPI Weak Scaling Analysis", x_label="MPI Ranks"):
    """
    Analyze and visualize weak scaling performance with Gustafson's Law.
    
    Args:
        data_frame: DataFrame containing 'cores' and 'time' columns
        plot_title: Title for the speedup plot
        x_label: Label for x-axis
    """
    # Weak scaling: work scales with processors
    # Scaled speedup = P * (T1 / Tp) - measures total work completed
    baseline_time = data_frame['time'].iloc[0]
    data_frame['scaled_speedup'] = data_frame['cores'] * (baseline_time / data_frame['time'])
    data_frame['efficiency'] = data_frame['scaled_speedup'] / data_frame['cores']
    
    # Gustafson's Law: S(n) = s + (1-s)*n
    def gustafson_model(n, serial_fraction):
        return serial_fraction + (1 - serial_fraction) * n
    
    # Prepare data for curve fitting
    num_procs = data_frame['cores'].values
    measured_scaled_speedup = data_frame['scaled_speedup'].values
    
    # Fit Gustafson's model
    fitted_params, _ = curve_fit(gustafson_model, num_procs, measured_scaled_speedup, bounds=(0, 1))
    serial_frac = fitted_params[0]
    predicted_scaled_speedup = gustafson_model(num_procs, serial_frac)
    
    # Scaled speedup visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(num_procs, measured_scaled_speedup, marker='o', markersize=9, linestyle='',
             color="#9C27B0", label='Measured Scaled Speedup', linewidth=2)
    ax1.plot(num_procs, num_procs, linestyle='-.', color='#4CAF50',
             label='Ideal Scaled Speedup', linewidth=2)
    ax1.plot(num_procs, predicted_scaled_speedup, linestyle='-', color='#FF9800',
             label=f"Gustafson Model (s={serial_frac:.3f})", linewidth=2)
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Scaled Speedup', fontsize=12, fontweight='bold')
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Efficiency visualization
    mean_eff = data_frame['efficiency'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_frame['cores'], data_frame['efficiency'], marker='o', markersize=9,
             color="#9C27B0", linestyle='None', label='Measured Efficiency', linewidth=2)
    ax2.axhline(y=1.0, color='#4CAF50', linestyle='-.', linewidth=2, label='Ideal Efficiency')
    ax2.axhline(y=mean_eff, color='#9E9E9E', linestyle='-', linewidth=2,
                label=f'Mean Efficiency ({mean_eff:.3f})')
    ax2.set_xlabel('MPI Rank Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.set_title('MPI Weak Scaling: Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
    return


def analyze_openmp_weak_scaling(data_frame, plot_title="OpenMP Weak Scaling Analysis", x_label="Thread Count"):
    """
    Analyze and visualize OpenMP weak scaling with Gustafson's Law.
    
    Args:
        data_frame: DataFrame containing 'threads' and 'time' columns
        plot_title: Title for the speedup plot
        x_label: Label for x-axis
    """
    # Weak scaling: work scales with threads
    # Scaled speedup = P * (T1 / Tp)
    baseline_time = data_frame['time'].iloc[0]
    data_frame['scaled_speedup'] = data_frame['threads'] * (baseline_time / data_frame['time'])
    data_frame['efficiency'] = data_frame['scaled_speedup'] / data_frame['threads']
    
    # Gustafson's Law: S(n) = s + (1-s)*n
    def gustafson_model(n, serial_fraction):
        return serial_fraction + (1 - serial_fraction) * n
    
    # Prepare data for curve fitting
    num_threads = data_frame['threads'].values
    measured_scaled_speedup = data_frame['scaled_speedup'].values
    
    # Fit Gustafson's model
    fitted_params, _ = curve_fit(gustafson_model, num_threads, measured_scaled_speedup, bounds=(0, 1))
    serial_frac = fitted_params[0]
    predicted_scaled_speedup = gustafson_model(num_threads, serial_frac)
    
    # Scaled speedup visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(num_threads, measured_scaled_speedup, marker='^', markersize=9, linestyle='',
             color="#00BCD4", label='Measured Scaled Speedup', linewidth=2)
    ax1.plot(num_threads, num_threads, linestyle='-.', color='#4CAF50',
             label='Ideal Scaled Speedup', linewidth=2)
    ax1.plot(num_threads, predicted_scaled_speedup, linestyle='-', color='#FF9800',
             label=f"Gustafson Model (s={serial_frac:.3f})", linewidth=2)
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Scaled Speedup', fontsize=12, fontweight='bold')
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Efficiency visualization
    mean_eff = data_frame['efficiency'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_frame['threads'], data_frame['efficiency'], marker='^', markersize=9,
             color="#00BCD4", linestyle='None', label='Measured Efficiency', linewidth=2)
    ax2.axhline(y=1.0, color='#4CAF50', linestyle='-.', linewidth=2, label='Ideal Efficiency')
    ax2.axhline(y=mean_eff, color='#9E9E9E', linestyle='-', linewidth=2,
                label=f'Mean Efficiency ({mean_eff:.3f})')
    ax2.set_xlabel('OpenMP Thread Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.set_title('OpenMP Weak Scaling: Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
    return