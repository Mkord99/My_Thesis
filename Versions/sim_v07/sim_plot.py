"""
Point Cloud 3D Visualizer with Parallel Processing

This script loads and visualizes multiple .xyz point cloud files from laser scanning simulations.

Features:
- Parallel processing using all CPU cores for faster loading
- Z-value filtering (removes points with z < 0.03m)
- Same color visualization for all point clouds  
- Point density heatmap based on spatial concentration (not intensity values)
- Two separate visualizations in different figures
- Progress tracking with detailed logging
- Performance timing and statistics

Requirements:
- numpy, pandas, matplotlib
- multiprocessing (built-in)

Usage:
1. Basic usage (parallel processing by default):
   python script.py
   
2. Custom usage:
   main_with_options(use_parallel=True, n_processes=4)   # Use 4 cores
   main_with_options(use_parallel=False)                 # Sequential processing

File Format Expected:
- Column 1: X coordinate
- Column 2: Y coordinate  
- Column 3: Z coordinate
- Column 4: Intensity (optional, for reference only)

Output:
- Figure 1: Regular point cloud visualization (all points in blue)
- Figure 2: Point density heatmap (color-coded by number of points per spatial cell)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, Lock, Value
import time
from functools import partial

# Global variables for progress tracking
progress_counter = 0
total_files_global = 0

def read_xyz_file_simple(filepath):
    """
    Simple XYZ file reader for parallel processing.
    Assumes format: x y z [intensity] [additional_columns]
    Filters out points with z < 0.03
    """
    filename = os.path.basename(filepath)
    
    try:
        # Try reading with pandas
        data = pd.read_csv(filepath, sep='\s+', header=None, engine='python')
        
        # Extract x, y, z coordinates and intensity if available
        if data.shape[1] >= 3:
            points = data.iloc[:, :3].values  # x, y, z
            intensity = None
            
            # Check if intensity column exists (4th column)
            if data.shape[1] >= 4:
                intensity = data.iloc[:, 3].values
            
            # Filter out points with z < 0.03
            z_filter = points[:, 2] >= 0.03
            points_filtered = points[z_filter]
            
            if intensity is not None:
                intensity_filtered = intensity[z_filter]
            else:
                intensity_filtered = None
            
            original_count = len(points)
            filtered_count = len(points_filtered)
            removed_count = original_count - filtered_count
            
            return (filename, points_filtered, intensity_filtered, filtered_count, True, removed_count)
            
        else:
            return (filename, None, None, 0, False, 0)
            
    except Exception as e:
        return (filename, None, None, 0, False, 0)

def progress_callback(result):
    """Callback function to track progress"""
    global progress_counter, total_files_global
    progress_counter += 1
    
    filename, points, intensity, num_points, success, removed_count = result
    progress_percent = (progress_counter / total_files_global) * 100
    
    if success:
        intensity_info = " (with intensity)" if intensity is not None else ""
        removed_info = f" (removed {removed_count} low-z points)" if removed_count > 0 else ""
        print(f"[{progress_counter:3d}/{total_files_global}] ({progress_percent:5.1f}%) ✓ {filename} - {num_points:,} points{intensity_info}{removed_info}")
    else:
        print(f"[{progress_counter:3d}/{total_files_global}] ({progress_percent:5.1f}%) ✗ {filename} - Failed to load")
    
    # Add visual separator every 10 files
    if progress_counter % 10 == 0 and progress_counter < total_files_global:
        print("-" * 50)

def read_xyz_file(filepath):
    """
    Read an XYZ point cloud file.
    Assumes format: x y z [intensity] [additional_columns]
    Filters out points with z < 0.03
    """
    try:
        # Try reading with pandas first (handles various separators automatically)
        data = pd.read_csv(filepath, sep='\s+', header=None, engine='python')
        
        # Extract x, y, z coordinates and intensity if available
        if data.shape[1] >= 3:
            points = data.iloc[:, :3].values  # x, y, z
            intensity = None
            
            # Check if intensity column exists (4th column)
            if data.shape[1] >= 4:
                intensity = data.iloc[:, 3].values
            
            # Filter out points with z < 0.03
            z_filter = points[:, 2] >= 0.03
            points_filtered = points[z_filter]
            
            if intensity is not None:
                intensity_filtered = intensity[z_filter]
            else:
                intensity_filtered = None
                
            return points_filtered, intensity_filtered
        else:
            print(f"Warning: {filepath} doesn't have enough columns (needs at least 3)")
            return None, None
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def load_all_point_clouds_parallel(directory_path, n_processes=None):
    """
    Load all .xyz files from the specified directory using parallel processing.
    Returns a list of tuples: (filename, point_cloud_array, intensity_array)
    
    Args:
        directory_path: Path to directory containing .xyz files
        n_processes: Number of processes to use (None = all CPU cores)
    """
    global progress_counter, total_files_global
    
    # Find all .xyz files in the directory
    xyz_pattern = os.path.join(directory_path, "*.xyz")
    xyz_files = glob.glob(xyz_pattern)
    
    if not xyz_files:
        print(f"No .xyz files found in {directory_path}")
        return []
    
    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    total_files = len(xyz_files)
    total_files_global = total_files
    progress_counter = 0
    
    print(f"Found {total_files} .xyz files")
    print(f"Using {n_processes} CPU cores for parallel processing")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    
    point_clouds = []
    
    try:
        print("🚀 Starting parallel file loading...")
        
        # Use multiprocessing Pool with proper context
        with mp.Pool(processes=n_processes) as pool:
            # Submit all jobs
            results = []
            for filepath in xyz_files:
                result = pool.apply_async(read_xyz_file_simple, (filepath,), callback=progress_callback)
                results.append(result)
            
            # Wait for all results and collect them
            for result in results:
                filename, points, intensity, num_points, success, removed_count = result.get()
                if success and points is not None and len(points) > 0:  # Only add non-empty point clouds
                    point_clouds.append((filename, points, intensity))
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("=" * 70)
        print(f"⚡ Parallel loading completed in {processing_time:.2f} seconds")
        print(f"📊 Processing speed: {total_files/processing_time:.1f} files/second")
        print(f"🎯 Successfully loaded {len(point_clouds)} out of {total_files} files")
        
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        print("🔄 Falling back to sequential processing...")
        return load_all_point_clouds(directory_path)
    
    return point_clouds

def load_all_point_clouds_parallel_v2(directory_path, n_processes=None):
    """
    Alternative parallel loading method using pool.map with chunks.
    More reliable on some systems.
    """
    # Find all .xyz files in the directory
    xyz_pattern = os.path.join(directory_path, "*.xyz")
    xyz_files = glob.glob(xyz_pattern)
    
    if not xyz_files:
        print(f"No .xyz files found in {directory_path}")
        return []
    
    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    total_files = len(xyz_files)
    print(f"Found {total_files} .xyz files")
    print(f"Using {n_processes} CPU cores for parallel processing (method 2)")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    
    point_clouds = []
    
    try:
        print("🚀 Starting parallel file loading (chunked method)...")
        
        # Calculate chunk size for better load balancing
        chunk_size = max(1, total_files // (n_processes * 4))
        
        # Use pool.map with chunking
        if __name__ == '__main__' or True:  # Force multiprocessing
            with mp.Pool(processes=n_processes) as pool:
                print(f"📦 Processing in chunks of {chunk_size} files")
                results = pool.map(read_xyz_file_simple, xyz_files, chunksize=chunk_size)
        
        # Process results and show progress
        for i, (filename, points, intensity, num_points, success, removed_count) in enumerate(results, 1):
            progress_percent = (i / total_files) * 100
            
            if success and points is not None and len(points) > 0:  # Only add non-empty point clouds
                point_clouds.append((filename, points, intensity))
                intensity_info = " (with intensity)" if intensity is not None else ""
                removed_info = f" (removed {removed_count} low-z points)" if removed_count > 0 else ""
                print(f"[{i:3d}/{total_files}] ({progress_percent:5.1f}%) ✓ {filename} - {num_points:,} points{intensity_info}{removed_info}")
            else:
                print(f"[{i:3d}/{total_files}] ({progress_percent:5.1f}%) ✗ {filename} - Failed to load or empty")
            
            # Add visual separator every 10 files
            if i % 10 == 0 and i < total_files:
                print("-" * 50)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("=" * 70)
        print(f"⚡ Parallel loading completed in {processing_time:.2f} seconds")
        print(f"📊 Processing speed: {total_files/processing_time:.1f} files/second")
        print(f"🎯 Successfully loaded {len(point_clouds)} out of {total_files} files")
        
    except Exception as e:
        print(f"❌ Error in parallel processing (method 2): {e}")
        print("🔄 Falling back to sequential processing...")
        return load_all_point_clouds(directory_path)
    
    return point_clouds

def test_multiprocessing():
    """Test if multiprocessing is working correctly"""
    def square_number(x):
        return x * x
    
    try:
        n_cores = mp.cpu_count()
        print(f"🧪 Testing multiprocessing with {n_cores} cores...")
        
        # Use a module-level function to avoid pickle issues
        with mp.Pool(processes=2) as pool:
            test_data = [1, 2, 3, 4]
            results = pool.map(square_number, test_data)
            
        if results == [1, 4, 9, 16]:
            print("✅ Multiprocessing test passed!")
            return True
        else:
            print("❌ Multiprocessing test failed - unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Multiprocessing test failed: {e}")
        return False

def square_number(x):
    """Helper function for multiprocessing test"""
    return x * x

def load_all_point_clouds(directory_path):
    """
    Load all .xyz files from the specified directory (sequential processing).
    Returns a list of tuples: (filename, point_cloud_array, intensity_array)
    """
    # Find all .xyz files in the directory
    xyz_pattern = os.path.join(directory_path, "*.xyz")
    xyz_files = glob.glob(xyz_pattern)
    
    if not xyz_files:
        print(f"No .xyz files found in {directory_path}")
        return []
    
    print(f"Found {len(xyz_files)} .xyz files")
    print("Using sequential processing")
    print("=" * 60)
    
    point_clouds = []
    total_files = len(xyz_files)
    start_time = time.time()
    
    for i, filepath in enumerate(xyz_files, 1):
        filename = os.path.basename(filepath)
        
        # Progress indicator
        progress_percent = (i / total_files) * 100
        print(f"[{i:3d}/{total_files}] ({progress_percent:5.1f}%) Loading {filename}...")
        
        points, intensity = read_xyz_file(filepath)
        if points is not None and len(points) > 0:  # Only add non-empty point clouds
            point_clouds.append((filename, points, intensity))
            intensity_info = f" (with intensity)" if intensity is not None else ""
            print(f"           ✓ Loaded {len(points):,} points{intensity_info} (z ≥ 0.03)")
        else:
            print(f"           ✗ Failed to load {filename} or empty after filtering")
        
        # Add a visual separator every 10 files for better readability
        if i % 10 == 0 and i < total_files:
            print("-" * 40)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print("=" * 60)
    print(f"Sequential loading completed in {processing_time:.2f} seconds")
    
    return point_clouds

def calculate_point_density(point_clouds, grid_size=0.1):
    """
    Calculate point density for all point clouds combined.
    Creates a 3D grid and counts points in each grid cell.
    
    Args:
        point_clouds: List of (filename, points, intensity) tuples
        grid_size: Size of grid cells for density calculation
    
    Returns:
        Dictionary with point coordinates as keys and density counts as values
    """
    print(f"🔢 Calculating point density with grid size {grid_size}m...")
    
    # Combine all points
    all_points = []
    for _, points, _ in point_clouds:
        if len(points) > 0:  # Only add non-empty point clouds
            all_points.append(points)
    
    if not all_points:
        print("   No points to calculate density")
        return {}
    
    combined_points = np.vstack(all_points)
    
    # Calculate grid bounds
    min_coords = np.min(combined_points, axis=0)
    max_coords = np.max(combined_points, axis=0)
    
    print(f"   Point cloud bounds: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], "
          f"Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
    
    # Create grid indices for each point
    grid_indices = np.floor((combined_points - min_coords) / grid_size).astype(int)
    
    # Count points in each grid cell
    from collections import defaultdict
    density_dict = defaultdict(int)
    
    for idx in grid_indices:
        key = tuple(idx)
        density_dict[key] += 1
    
    # Convert back to point coordinates with density values
    point_density = {}
    for grid_key, count in density_dict.items():
        # Calculate center of grid cell
        center_coords = min_coords + (np.array(grid_key) + 0.5) * grid_size
        point_density[tuple(center_coords)] = count
    
    print(f"   Created density grid with {len(point_density)} occupied cells")
    if density_dict:
        print(f"   Density range: {min(density_dict.values())} to {max(density_dict.values())} points per cell")
    
    return point_density

def create_density_arrays(point_clouds, grid_size=0.1):
    """
    Create arrays for density visualization.
    
    Returns:
        coords: Nx3 array of grid center coordinates
        densities: N array of density values
    """
    density_dict = calculate_point_density(point_clouds, grid_size)
    
    if not density_dict:
        return np.array([]), np.array([])
    
    coords = np.array(list(density_dict.keys()))
    densities = np.array(list(density_dict.values()))
    
    return coords, densities

def plot_point_clouds_3d(point_clouds, max_points_per_cloud=10000):
    """
    Plot all point clouds in separate 3D figures:
    1. Regular point cloud visualization (same color)
    2. Point density heatmap (based on number of points per location)
    
    Args:
        point_clouds: List of tuples (filename, points_array, intensity_array)
        max_points_per_cloud: Maximum number of points to plot per cloud (for performance)
    """
    if not point_clouds:
        print("No point clouds to plot")
        return
    
    # Filter out empty point clouds
    non_empty_clouds = [(filename, points, intensity) for filename, points, intensity in point_clouds if len(points) > 0]
    
    if not non_empty_clouds:
        print("No non-empty point clouds to plot")
        return
    
    print("🎨 Creating visualization 1: Regular Point Cloud (same color)")
    
    # === FIGURE 1: Regular Point Cloud ===
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    total_points_plotted = 0
    
    # Plot each point cloud with same color
    for i, (filename, points, intensity) in enumerate(non_empty_clouds):
        # Subsample points if there are too many (for better performance)
        if len(points) > max_points_per_cloud:
            indices = np.random.choice(len(points), max_points_per_cloud, replace=False)
            points_to_plot = points[indices]
            print(f"  Subsampling {filename}: {len(points):,} -> {max_points_per_cloud:,} points")
        else:
            points_to_plot = points
        
        # Plot with same blue color
        ax1.scatter(points_to_plot[:, 0], 
                   points_to_plot[:, 1], 
                   points_to_plot[:, 2],
                   c='blue', 
                   s=1,  # Point size
                   alpha=0.6)
        
        total_points_plotted += len(points_to_plot)
    
    # Set labels and title for figure 1
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Point Cloud Visualization (z ≥ 0.03m)\n{len(non_empty_clouds)} files, {total_points_plotted:,} points plotted')
    
    # Set equal aspect ratio for figure 1
    if non_empty_clouds:
        max_range = 0
        for _, points, _ in non_empty_clouds:
            if len(points) > 0:  # Double check for non-empty
                ranges = [points[:, i].max() - points[:, i].min() for i in range(3)]
                max_range = max(max_range, max(ranges))
        
        # Center the plot
        all_points = np.vstack([points for _, points, _ in non_empty_clouds if len(points) > 0])
        center = [np.mean(all_points[:, i]) for i in range(3)]
        
        ax1.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax1.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax1.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    plt.tight_layout()
    plt.show()
    
    # === FIGURE 2: Point Density Heatmap ===
    print("🌈 Creating visualization 2: Point Density Heatmap")
    
    # Calculate point density
    density_coords, density_values = create_density_arrays(non_empty_clouds, grid_size=0.05)  # 5cm grid
    
    if len(density_coords) == 0:
        print("❌ No density data to plot")
        return
    
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Create density heatmap
    scatter = ax2.scatter(density_coords[:, 0], 
                         density_coords[:, 1], 
                         density_coords[:, 2],
                         c=density_values,
                         cmap='viridis',  # Color map
                         s=20,  # Larger points for density visualization
                         alpha=0.8,
                         vmin=np.min(density_values),
                         vmax=np.max(density_values))
    
    # Add colorbar for density
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, aspect=20)
    cbar.set_label('Point Density (points per 5cm³ cell)', rotation=270, labelpad=20)
    
    # Set labels and title for figure 2
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Point Density Heatmap (z ≥ 0.03m)\n{len(density_coords):,} grid cells, density range: {np.min(density_values)}-{np.max(density_values)} points/cell')
    
    # Use same axis limits as figure 1
    if non_empty_clouds:
        ax2.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax2.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax2.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    plt.tight_layout()
    plt.show()

def plot_individual_clouds(point_clouds, max_points_per_cloud=10000):
    """
    Plot each point cloud in a separate subplot.
    """
    if not point_clouds:
        print("No point clouds to plot")
        return
    
    # Filter out empty point clouds
    non_empty_clouds = [(filename, points, intensity) for filename, points, intensity in point_clouds if len(points) > 0]
    
    if not non_empty_clouds:
        print("No non-empty point clouds to plot")
        return
    
    # Determine subplot grid
    n_clouds = len(non_empty_clouds)
    cols = min(4, n_clouds)
    rows = (n_clouds + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    for i, (filename, points, intensity) in enumerate(non_empty_clouds):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Subsample if necessary
        if len(points) > max_points_per_cloud:
            indices = np.random.choice(len(points), max_points_per_cloud, replace=False)
            points_to_plot = points[indices]
        else:
            points_to_plot = points
        
        # Plot
        ax.scatter(points_to_plot[:, 0], 
                  points_to_plot[:, 1], 
                  points_to_plot[:, 2],
                  c='blue',
                  s=1, alpha=0.6)
        
        intensity_info = " (with intensity)" if intensity is not None else ""
        ax.set_title(f"{filename}\n{len(points):,} points{intensity_info} (z ≥ 0.03m)")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Directory containing the .xyz files
    directory_path = "/home/mo/thesis/My_Thesis/Versions/sim_v07/output/buildings_walking_person_custom/2025-06-24_00-35-57" 
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        print("Please check the path and try again.")
        return
    
    print(f"Loading point clouds from: {directory_path}")
    print("=" * 80)
    
    # CPU information and multiprocessing test
    n_cores = mp.cpu_count()
    print(f"💻 System has {n_cores} CPU cores available")
    
    # Test multiprocessing
    mp_working = test_multiprocessing()
    
    # Choose processing method
    if mp_working:
        print("🚀 Multiprocessing is working! Using parallel processing...")
        
        # Try method 2 first (more reliable)
        print("🔄 Trying parallel processing method 2...")
        point_clouds = load_all_point_clouds_parallel_v2(directory_path)
        
        # If that fails, try method 1
        if not point_clouds:
            print("🔄 Trying parallel processing method 1...")
            point_clouds = load_all_point_clouds_parallel(directory_path)
            
        # If both fail, use sequential
        if not point_clouds:
            print("🔄 Both parallel methods failed, using sequential...")
            point_clouds = load_all_point_clouds(directory_path)
    else:
        print("⚠️  Multiprocessing not working properly, using sequential processing...")
        point_clouds = load_all_point_clouds(directory_path)
    
    if not point_clouds:
        print("❌ No valid point clouds loaded!")
        return
    
    print(f"\n✅ Successfully loaded {len(point_clouds)} point clouds")
    
    # Calculate total points and check for intensity data
    total_points = sum(len(points) for _, points, _ in point_clouds)
    has_intensity_data = any(intensity is not None for _, _, intensity in point_clouds)
    intensity_files = sum(1 for _, _, intensity in point_clouds if intensity is not None)
    
    print(f"📊 Total points across all clouds: {total_points:,} (after z ≥ 0.03m filter)")
    if has_intensity_data:
        print(f"🎯 Found intensity data in {intensity_files}/{len(point_clouds)} files")
    else:
        print("ℹ️  No intensity data found (files have only X, Y, Z coordinates)")
    
    # Show file statistics
    print(f"\n📁 File Statistics:")
    print("-" * 40)
    for filename, points, intensity in point_clouds[:5]:  # Show first 5 files
        intensity_info = " + intensity" if intensity is not None else ""
        print(f"  {filename}: {len(points):,} points{intensity_info}")
    
    if len(point_clouds) > 5:
        print(f"  ... and {len(point_clouds) - 5} more files")
    
    print("\n" + "=" * 80)
    print("🎨 CREATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create the main visualizations
    print("\n🎨 Creating visualizations:")
    print("   1. Regular point cloud (same blue color)")
    print("   2. Point density heatmap (based on spatial point concentration)")
    
    plot_point_clouds_3d(point_clouds)
    
    # If there are few files, also show individual plots
    if len(point_clouds) <= 12:
        print("\n🎨 Creating individual plots...")
        plot_individual_clouds(point_clouds)
    else:
        print(f"\nℹ️  Skipping individual plots (too many files: {len(point_clouds)})")
        print("   Use plot_individual_clouds() function if needed")
    
    print("\n🎉 Visualization complete!")

# Alternative function to easily switch between processing modes
def main_with_options(use_parallel=True, n_processes=None, parallel_method=2):
    """
    Main function with options for processing mode.
    
    Args:
        use_parallel: True for parallel processing, False for sequential
        n_processes: Number of processes (None = all cores)
        parallel_method: 1 or 2 (different parallel approaches)
    """
    # Directory containing the .xyz files
    directory_path = "/home/mo/thesis/My_Thesis/Versions/sim_v07/output/buildings_walking_person_custom/2025-06-24_00-35-57"
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        print("Please check the path and try again.")
        return
    
    print(f"Loading point clouds from: {directory_path}")
    print("=" * 80)
    
    # CPU information
    n_cores = mp.cpu_count()
    print(f"💻 System has {n_cores} CPU cores available")
    
    if use_parallel:
        if n_processes is None:
            n_processes = n_cores
            
        print(f"🚀 Using parallel processing with {n_processes} cores (method {parallel_method})...")
        
        if parallel_method == 2:
            point_clouds = load_all_point_clouds_parallel_v2(directory_path, n_processes)
        else:
            point_clouds = load_all_point_clouds_parallel(directory_path, n_processes)
    else:
        print("🔄 Using sequential processing...")
        point_clouds = load_all_point_clouds(directory_path)
    
    if not point_clouds:
        return
    
    # Rest of the processing...
    print(f"\n✅ Successfully loaded {len(point_clouds)} point clouds")
    
    # Calculate total points and check for intensity data
    total_points = sum(len(points) for _, points, _ in point_clouds)
    has_intensity_data = any(intensity is not None for _, _, intensity in point_clouds)
    intensity_files = sum(1 for _, _, intensity in point_clouds if intensity is not None)
    
    print(f"📊 Total points across all clouds: {total_points:,} (after z ≥ 0.03m filter)")
    if has_intensity_data:
        print(f"🎯 Found intensity data in {intensity_files}/{len(point_clouds)} files")
    
    # Create visualizations
    plot_point_clouds_3d(point_clouds)
    
    if len(point_clouds) <= 12:
        plot_individual_clouds(point_clouds)
    
    print("\n🎉 Visualization complete!")

if __name__ == "__main__":
    main()