import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

def build_intercore_latency_matrix(n_cores=256):
    """
    Construct latency matrix for inter-core communication based on topology.
    
    Args:
        n_cores: Total number of cores in the system
        
    Returns:
        numpy.ndarray: Square matrix of communication latencies in seconds
    """
    # Empirical latency measurements (in seconds)
    LAT_CCX = 0.14e-6          # Same CCX 
    LAT_CCD = 0.32e-6          # Different CCX, same CCD
    LAT_NUMA = 0.35e-6         # Same NUMA domain
    LAT_SOCKET = 0.37e-6        # Different NUMA, same socket 
    LAT_CROSS_SOCKET = 0.65e-6  # Different socket, same node 
    LAT_CROSS_NODE = 1.82e-6     # Different node

    latencies = np.zeros((n_cores, n_cores), dtype=np.float32)
    
    for src in range(n_cores):
        for dst in range(n_cores):
            if src == dst:
                latencies[src][dst] = 0
            elif src // 4 == dst // 4:
                latencies[src][dst] = LAT_CCX
            elif src // 16 == dst // 16:
                latencies[src][dst] = LAT_CCD
            elif src // 32 == dst // 32:
                latencies[src][dst] = LAT_NUMA
            elif src // 64 == dst // 64:
                latencies[src][dst] = LAT_SOCKET
            elif src // 128 == dst // 128:
                latencies[src][dst] = LAT_CROSS_SOCKET
            else:
                latencies[src][dst] = LAT_CROSS_NODE
                
    return latencies

# PLOT 

def theory_and_experiment_plot(data_exp, data_theory, title_name, position, factor=1):
    
    plt.figure(figsize=(14, 8))

    # Extract experimental data
    experimental_processes = data_exp['idx_process']
    experimental_latency = data_exp['Latency']*factor

    # Extract model predictions
    model_processes = list(data_theory.keys())
    model_latency = list(data_theory.values())
    model_latency = [factor*j for j in model_latency]

    # Plot experimental results
    plt.plot(experimental_processes, experimental_latency, 
            marker='o', linestyle='--', linewidth=2,
            label='Experimental Data', 
            color='#2E86AB', markersize=6, markerfacecolor='white', 
            markeredgewidth=2.5, alpha=0.85)

    # Plot model predictions
    plt.plot(model_processes, model_latency, 
            linewidth=3, label='Analytical Model', 
            color='#A23B72', alpha=0.9)

    # Hardware topology boundaries with larger labels
    topology_markers = [
        (1, 'Core 1\nSame CCX', '#E8E8E8'),
        (4, 'Core 4\nSame CCD', '#D0D0D0'),
        (8, 'Core 8\nSame NUMA', '#B8B8B8'),
        (32, 'Core 32\nCross NUMA', '#A0A0A0'),
        (64, 'Core 64\nCross Socket', '#888888'),
        (128, 'Core 128\nInter-Node', '#606060')
    ]

    for core_id, label, color in topology_markers:
        plt.axvline(x=core_id, color=color, linestyle=':', linewidth=2.5, alpha=0.7)
        plt.text(core_id, plt.ylim()[1] * 0.97, label, 
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=11, fontweight='bold', alpha=0.8, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

    # Styling
    plt.xlabel('Number of Processes', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (μs)', fontsize=14, fontweight='bold')
    plt.title(title_name + ': Model vs Experimental Performance', 
            fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc=position, fontsize=12, framealpha=0.95, 
            edgecolor='gray', fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    plt.gca().set_facecolor('#FAFAFA')

    # Add subtle frame
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()


def plot_errors(data_exp,  data_theory, overtitle, position1, position2):
    # Create comprehensive error analysis visualization
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(overtitle+': Model Error Analysis', fontsize=16, fontweight='bold', y=1.00)

    # Extract experimental data
    experimental_processes = data_exp['idx_process'].values
    experimental_latency = data_exp['Latency'].values

    # Extract corresponding model predictions
    model_latency_aligned = [data_theory[n] for n in experimental_processes]

    # Calculate errors
    absolute_error = [model - exp for model, exp in zip(model_latency_aligned, experimental_latency)]
    relative_error = [(model - exp) / exp * 100 for model, exp in zip(model_latency_aligned, experimental_latency)]

    # ===========================
    # Plot 0: Absolute Error
    # ===========================
    ax0.plot(experimental_processes, absolute_error, 
            marker='s', linestyle='-', linewidth=2.5,
            color='#E63946', markersize=7, markerfacecolor='white', 
            markeredgewidth=2, alpha=0.85, label='Absolute Error')

    # Zero reference line
    ax0.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Perfect Match')

    # Hardware topology boundaries
    topology_markers = [
        (1, 'Same CCX', '#E8E8E8'),
        (4, 'Same CCD', '#D0D0D0'),
        (8, 'Same NUMA', '#B8B8B8'),
        (32, 'Cross NUMA', '#A0A0A0'),
        (64, 'Cross Socket', '#888888'),
        (128, 'Inter-Node', '#606060')
    ]

    for core_id, label, color in topology_markers:
        ax0.axvline(x=core_id, color=color, linestyle=':', linewidth=2, alpha=0.6)
        ax0.text(core_id, ax0.get_ylim()[1] * 0.95, label, 
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=10, fontweight='bold', alpha=0.75, color=color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.7, edgecolor=color))

    # Styling
    ax0.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax0.set_ylabel('Absolute Error (μs)', fontsize=13, fontweight='bold')
    ax0.set_title('Absolute Error: Model - Experimental', fontsize=14, fontweight='bold', pad=15)
    ax0.legend(loc=position1, fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax0.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax0.set_facecolor('#FAFAFA')



    # ===========================
    # Plot 1: Relative Error (%)
    # ===========================
    ax1.plot(experimental_processes, relative_error, 
            marker='o', linestyle='-', linewidth=2.5,
            color='#F77F00', markersize=7, markerfacecolor='white', 
            markeredgewidth=2, alpha=0.85, label='Relative Error')

    # Zero reference line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Perfect Match')

    # Acceptable error bands (±5%, ±10%)
    ax1.axhspan(-5, 5, alpha=0.15, color='green', label='±5% tolerance')
    ax1.axhspan(-10, 10, alpha=0.1, color='yellow', label='±10% tolerance')

    # Hardware topology boundaries
    for core_id, label, color in topology_markers:
        ax1.axvline(x=core_id, color=color, linestyle=':', linewidth=2, alpha=0.6)
        ax1.text(core_id, ax1.get_ylim()[1] * 0.95, label, 
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=10, fontweight='bold', alpha=0.75, color=color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.7, edgecolor=color))

    # Styling
    ax1.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Relative Error (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Relative Error: (Model - Experimental) / Experimental × 100', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc=position2, fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax1.set_facecolor('#FAFAFA')



    # Frame styling for both plots
    for ax in [ax0, ax1]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()




def map3d_grid(df, title="3D Surface Plot"):
    """
    Create a 3D surface plot for MPI broadcast latency data
    
    Parameters:
    df: DataFrame with columns 'number_processes', 'Size', 'Latency'
    title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    processes = df['number_processes'].values
    sizes = df['Size'].values
    latency = df['Latency'].values
    
    # Create grid for interpolation
    processes_unique = np.sort(df['number_processes'].unique())
    sizes_unique = np.sort(df['Size'].unique())
    
    # Create meshgrid
    P, S = np.meshgrid(processes_unique, sizes_unique)
    
    # Try different interpolation methods if cubic fails
    try:
        L = griddata((processes, sizes), latency, (P, S), method='cubic')
        # Fill NaN values with linear interpolation
        if np.isnan(L).any():
            L_linear = griddata((processes, sizes), latency, (P, S), method='linear')
            mask = np.isnan(L)
            L[mask] = L_linear[mask]
    except:
        # Fallback to linear interpolation
        L = griddata((processes, sizes), latency, (P, S), method='linear')
    
    # Only plot surface if we have valid interpolated data
    if not np.isnan(L).all():
        surf = ax.plot_surface(P, S, L, cmap=cm.plasma, alpha=0.7, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Latency (μs)')
    
    # Add thin lines connecting data points for texture
    for proc in processes_unique:
        proc_mask = processes == proc
        if np.sum(proc_mask) > 1:
            proc_sizes = sizes[proc_mask]
            proc_latency = latency[proc_mask]
            sort_idx = np.argsort(proc_sizes)
            ax.plot(np.full(len(sort_idx), proc), proc_sizes[sort_idx], proc_latency[sort_idx], 
                   color='navy', alpha=0.5, linewidth=1.0)
    
    for size in sizes_unique:
        size_mask = sizes == size
        if np.sum(size_mask) > 1:
            size_processes = processes[size_mask]
            size_latency = latency[size_mask]
            sort_idx = np.argsort(size_processes)
            ax.plot(size_processes[sort_idx], np.full(len(sort_idx), size), size_latency[sort_idx], 
                   color='navy', alpha=0.5, linewidth=1.0)
    
    # Add scatter points for actual measurements
    ax.scatter(processes, sizes, latency, 
              c='cyan',
              s=80,
              alpha=0.9, 
              edgecolor='darkblue',
              linewidth=1.5,
              marker='o',
              depthshade=True)
    
    # Labels and title
    ax.set_xlabel('Process Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Message Size (Bytes)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Latency (μs)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Better viewing angle - adjusted for clearer view
    ax.view_init(elev=30, azim=135)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# THEORETICAL MODEL
def compute_linear_broadcast_time(comm_latency_matrix, process_count, source_rank, processing_overhead):
    """
    Compute total communication time for linear broadcast algorithm. (Refernce in project Algo1)
    
    In linear broadcast, the source process sends data to all other processes.
    The algorithm uses non-blocking sends where transmissions can overlap,
    so the bottleneck is determined by the slowest communication path.
    
    Formula: T = max(T_source→i for all i≠source) + α·n
    
    :param comm_latency_matrix: 2D array of point-to-point communication times (seconds)
    :param process_count: Number of participating processes
    :param source_rank: Rank of the broadcasting process (default=0)
    :param processing_overhead: Per-process overhead for message handling (seconds, default=0.15 μs)
    :return: Total broadcast time in seconds
    """
    if process_count <= 1:
        return 0.0  # Single process requires no communication
    
    # Gather communication times from source to all destination processes
    transmission_times = [
        comm_latency_matrix[source_rank][dest] 
        for dest in range(process_count) 
        if dest != source_rank
    ]
    
    # Bottleneck: the slowest transmission determines overall communication time
    critical_path_latency = max(transmission_times) if transmission_times else 0.0
    
    # Aggregate processing overhead across all processes
    cumulative_overhead = processing_overhead * process_count
    
    # Total time: critical path + overhead
    total_time = critical_path_latency + cumulative_overhead
    
    return total_time

def compute_chain_broadcast_latency(comm_matrix, process_count, source_rank):
    """
    Compute total communication latency for chain (pipeline) broadcast algorithm.
    
    In chain broadcast, data flows sequentially through a linear pipeline:
    Process 0 → Process 1 → Process 2 → ... → Process (n-1)
    
    Each process receives data from its predecessor and forwards to its successor.
    Total time is the sum of all hop-to-hop communication times.
    
    Formula: T = Σ T_{i→(i+1)} for i = 0 to n-2
    
    :param comm_matrix: 2D array of point-to-point communication latencies (seconds)
    :param process_count: Total number of participating processes
    :param source_rank: Rank of the originating process (default=0, currently unused)
    :return: Total pipeline broadcast latency in seconds
    """
    if process_count <= 1:
        return 0.0  # Single process requires no communication
    
    # Accumulate communication time for each sequential hop in the chain
    cumulative_latency = 0.0
    
    # Sum latencies from each process to its immediate successor
    for current_rank in range(process_count - 1):
        next_rank = current_rank + 1
        hop_latency = comm_matrix[current_rank][next_rank]
        cumulative_latency += hop_latency
    
    return cumulative_latency


def compute_binary_tree_broadcast_latency(comm_matrix, process_count, processing_overhead):
    """
    Calculate latency for binary tree broadcast algorithm with overhead consideration.
    In binary tree, the message flows in a tree structure with height log2(n).
    
    :param comm_matrix: Communication time matrix (2D numpy array) in seconds
    :param process_count: Total number of cores used
    :param root: Root process rank (default=0)
    :param processing_overhead: Overhead per core in seconds (default=0.15 μs)
    :return: Total broadcast latency in seconds
    """
    if process_count <= 1:
        return 0.0  # No latency if there's only one or no processes
    
    # Calculate tree height (number of levels)
    tree_height = int(np.ceil(np.log2(process_count)))
    
    # For binary tree, the latency is determined by the critical path
    # which is the path from root to the deepest leaf
    max_latency = 0.0
    
    # Calculate latency for each level of the tree
    for level in range(tree_height):
        # Number of nodes at this level
        nodes_at_level = min(2**level, process_count - (2**level - 1))
        if nodes_at_level <= 0:
            break
            
        # Find maximum communication time at this level
        level_max_time = 0.0
        
        # For each node at this level, find the communication time to its parent
        for node in range(nodes_at_level):
            node_id = (2**level - 1) + node
            if node_id >= process_count:
                break
                
            if level > 0:  # Not the root
                parent_id = (node_id - 1) // 2
                comm_time = comm_matrix[parent_id][node_id]
                level_max_time = max(level_max_time, comm_time)
        
        # Add this level's maximum time to total latency
        max_latency += level_max_time
    
    # Add overhead that scales with number of cores
    total_overhead = processing_overhead * process_count
    
    # Total latency is the critical path plus overhead
    total_latency = max_latency + total_overhead
    
    return total_latency

def binary_tree_reduce_latency(matrix_times, num_cores):
    """
    Binary Tree Reduce: At each level, multiple pairs communicate in parallel.
    Takes the max latency at each level (bottleneck).
    
    Formula: T_binary(P) = sum over levels of max{latency(r + 2^i -> r)}
    """
    if num_cores <= 1:
        return 0
    
    total_latency = 0
    num_levels = int(math.log2(num_cores))
    
    for level in range(num_levels):
        step_size = 2 ** level
        level_latencies = []
        
        # At each level, find all communicating pairs
        for receiver_rank in range(0, num_cores, 2 * step_size):
            sender_rank = receiver_rank + step_size
            if sender_rank < num_cores:
                level_latencies.append(matrix_times[sender_rank][receiver_rank])
        
        # The slowest communication at this level dominates
        if level_latencies:
            total_latency += max(level_latencies)
    
    return total_latency


def binomial_tree_reduce_latency(matrix_times, num_cores):
    """
    Binomial Tree Reduce: At each step i, rank 2^i sends directly to root (rank 0).
    Sequential communications to the root.
    
    Formula: T_binomial(P) = sum of latency(2^i -> 0)
    """
    if num_cores <= 1:
        return 0
    
    total_latency = 0
    num_steps = int(math.log2(num_cores))
    
    for step in range(num_steps):
        sender_rank = 2 ** step
        root_rank = 0
        total_latency += matrix_times[sender_rank][root_rank]
    
    return total_latency



def rabenseifner_reduce_latency(matrix_times, num_cores):
    """
    Rabenseifner's Reduce: Two-phase algorithm
    Phase 1: Reduce-Scatter (recursive halving)
    Phase 2: Allgather (recursive doubling/binomial tree)
    
    At each phase level, takes max latency across parallel communications.
    """
    if num_cores <= 1:
        return 0
    
    total_latency = 0
    num_levels = int(math.log2(num_cores))
    
    # Phase 1: Reduce-Scatter (recursive halving)
    for level in range(num_levels):
        step_size = 2 ** level
        phase1_latencies = []
        
        for rank in range(0, num_cores, 2 * step_size):
            partner_rank = rank + step_size
            if partner_rank < num_cores:
                # Bidirectional communication, use one direction as proxy
                phase1_latencies.append(matrix_times[rank][partner_rank])
        
        if phase1_latencies:
            total_latency += max(phase1_latencies)
    
    # Phase 2: Allgather (binomial tree gather)
    for level in range(num_levels):
        step_size = 2 ** level
        phase2_latencies = []
        
        for rank in range(0, num_cores, 2 * step_size):
            sender_rank = rank + step_size
            receiver_rank = rank
            if sender_rank < num_cores:
                phase2_latencies.append(matrix_times[sender_rank][receiver_rank])
        
        if phase2_latencies:
            total_latency += max(phase2_latencies)
    
    return total_latency