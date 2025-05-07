#!/usr/bin/env python3
"""
Main entry point for the visibility path planning application.
"""
import os
import json
import time
import logging
import signal
import sys
import gc
from datetime import datetime

from src.data_handler import GeometryLoader
from src.graph_builder import GraphBuilder
from src.visibility_analyzer import VisibilityAnalyzer
from src.optimizer import PathOptimizer
from src.visualizer import PathVisualizer
from src.utils import setup_logging, log_memory_usage
from src.rotation_utils import rotate_path

# Global variables to store state in case of interruption
G = None
grid_points = None
building = None
obstacles = None
segments = None
segment_visibility = None
edge_visibility = None
vrf = None
selected_edges = []
config = None
visualizer = None
model = None
times = {}
rotation_angle = 0
rotation_center = (0, 0)
rotation_enabled = False
geometry_loader = None

def cleanup():
    """Release memory and resources."""
    global G, grid_points, building, obstacles, segments, segment_visibility, edge_visibility, vrf, selected_edges, model
    
    logger = logging.getLogger(__name__)
    log_memory_usage(logger, "Before cleanup")
    
    # Clear large data structures
    G = None
    grid_points = None
    building = None
    obstacles = None
    segments = None
    segment_visibility = None
    edge_visibility = None
    vrf = None
    selected_edges = []
    
    # Force garbage collection
    gc.collect()
    
    log_memory_usage(logger, "After cleanup")

def signal_handler(sig, frame):
    """Handle interrupt signals and create a plot with current results."""
    logger = logging.getLogger(__name__)
    logger.warning("Received interrupt signal, stopping gracefully and showing current results")
    
    # Calculate metrics if we have data
    if G is not None and selected_edges:
        path_metrics = calculate_path_metrics(G, selected_edges, segments, segment_visibility)
        
        # Print metrics to terminal
        print("\n--- INTERRUPTED RESULTS ---")
        print(f"Path Length: {path_metrics['path_length']:.2f} units")
        print(f"Number of Edges: {path_metrics['num_edges']}")
        print(f"Visibility Ratio: {path_metrics['vrf']:.4f}")
        for step, step_time in times.items():
            print(f"{step} Time: {step_time:.2f} seconds")
        
        # Plot current results
        if visualizer is not None and building is not None and obstacles is not None and segments is not None:
            visualizer.plot(G, building, obstacles, segments, selected_edges, path_metrics, 
                           rotation_angle if rotation_enabled else None, 
                           rotation_center if rotation_enabled else None)
    
    # Clean up resources before exit
    cleanup()
    
    sys.exit(0)

def calculate_path_metrics(G, selected_edges, segments, segment_visibility):
    """
    Calculate metrics for the selected path.
    
    Args:
        G: networkx DiGraph
        selected_edges: List of selected edges
        segments: List of segments
        segment_visibility: Dictionary mapping segments to visible edges
        
    Returns:
        Dictionary with path metrics
    """
    metrics = {}
    
    # Calculate path length
    path_length = sum(G[i][j]['weight'] for i, j in selected_edges)
    metrics['path_length'] = path_length
    
    # Count number of edges
    metrics['num_edges'] = len(selected_edges)
    
    # Calculate Visibility Ratio Factor (VRF)
    # Count all segment visibilities from the selected edges (with duplicates)
    total_visible_segments = 0
    for edge in selected_edges:
        # Get all segments visible from this edge
        visible_from_edge = 0
        for seg_idx, edges in segment_visibility.items():
            if edge in edges:
                visible_from_edge += 1
        total_visible_segments += visible_from_edge
    
    metrics['total_visible_segments'] = total_visible_segments
    metrics['total_segments'] = len(segments)
    metrics['vrf'] = total_visible_segments / path_length if path_length > 0 else 0
    
    return metrics

def create_output_directories(config):
    """Create all necessary output directories."""
    # Create log directory
    if config['output']['logs']['enabled']:
        os.makedirs(config['output']['logs']['path'], exist_ok=True)
    
    # Create plots directory
    if config['output']['plots']['save']:
        os.makedirs(config['output']['plots']['path'], exist_ok=True)
    
    # Create visibility data directory
    visibility_dir = os.path.join("output", "visibility")
    os.makedirs(visibility_dir, exist_ok=True)
    
    # Create path data directory
    path_dir = os.path.join("output", "path")
    os.makedirs(path_dir, exist_ok=True)
    
    # Create orientation data directory if rotation is enabled
    if config.get('rotation', {}).get('enabled', False):
        orientation_dir = os.path.join("output", "orientation")
        os.makedirs(orientation_dir, exist_ok=True)

def save_path_to_file(G, selected_edges, rotation_angle, rotation_center, rotation_enabled):
    """
    Save the path to a file in both rotated and original coordinate systems if rotation is enabled.
    
    Args:
        G: networkx DiGraph
        selected_edges: List of selected edges
        rotation_angle: Rotation angle in degrees
        rotation_center: Tuple of (center_x, center_y)
        rotation_enabled: Whether rotation preprocessing was enabled
    """
    logger = logging.getLogger(__name__)
    
    if not selected_edges:
        logger.warning("No path to save")
        return
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join("output", "path")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract path nodes
    path_nodes = []
    for i, j in selected_edges:
        if not path_nodes:
            path_nodes.append(i)
        path_nodes.append(j)
    
    # Get coordinates for each node
    path_coords = []
    for node in path_nodes:
        pos = G.nodes[node]['pos']
        path_coords.append((pos[0], pos[1]))
    
    # If rotation is enabled, save both rotated and original coordinates
    if rotation_enabled:
        # Save rotated path
        with open(os.path.join(output_dir, "path_rotated.txt"), 'w') as f:
            f.write("# Path coordinates in rotated coordinate system\n")
            f.write("# x, y\n")
            for x, y in path_coords:
                f.write(f"{x:.4f}, {y:.4f}\n")
        
        # Reverse rotate coordinates to original system
        path_coords_original = rotate_path(path_coords, rotation_angle, rotation_center, inverse=True)
        
        # Save original path
        with open(os.path.join(output_dir, "path_original.txt"), 'w') as f:
            f.write("# Path coordinates in original coordinate system\n")
            f.write("# x, y\n")
            for x, y in path_coords_original:
                f.write(f"{x:.4f}, {y:.4f}\n")
        
        logger.info(f"Saved path coordinates to {output_dir}/path_rotated.txt and {output_dir}/path_original.txt")
    else:
        # Save only original path if rotation wasn't applied
        with open(os.path.join(output_dir, "path.txt"), 'w') as f:
            f.write("# Path coordinates\n")
            f.write("# x, y\n")
            for x, y in path_coords:
                f.write(f"{x:.4f}, {y:.4f}\n")
        
        logger.info(f"Saved path coordinates to {output_dir}/path.txt")

def print_rotation_debug_info(debug_info):
    """
    Print rotation debug information to the console.
    
    Args:
        debug_info: Debug information dictionary
    """
    if not debug_info:
        return
    
    print("\n--- ROTATION DEBUG INFORMATION ---")
    
    # Print edge information before rotation
    if 'edge_info' in debug_info:
        edge_info = debug_info['edge_info']
        print(f"Before rotation:")
        print(f"  Longest edge length: {edge_info['longest_edge_length']:.2f}m")
        print(f"  Angle with north: {edge_info['longest_edge_angle']:.2f}°")
        print(f"  Target angle: {edge_info['target_angle']}°")
        print(f"  Rotation needed: {edge_info['rotation_angle']:.2f}°")
    
    # Print verification information after rotation
    if 'verification' in debug_info:
        verify = debug_info['verification']
        print("\nAfter rotation:")
        print(f"  Original angle: {verify['original_angle']:.2f}°")
        print(f"  Rotated angle: {verify['rotated_angle']:.2f}°")
        print(f"  Target angle: {verify['target_angle']}°")
        print(f"  Rotation error: {verify['error']:.4f}°")
    
    # Print information about top edges if available
    if 'edges' in debug_info:
        print("\nTop 5 longest edges:")
        edges = sorted(debug_info['edges'], key=lambda x: x['length'], reverse=True)[:5]
        for i, edge in enumerate(edges):
            print(f"  {i+1}. Length: {edge['length']:.2f}m, Angle: {edge['angle']:.2f}°")
    
    print()

def main():
    global G, grid_points, building, obstacles, segments, segment_visibility, edge_visibility 
    global vrf, selected_edges, config, visualizer, model, times
    global rotation_angle, rotation_center, rotation_enabled, geometry_loader
    
    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Check if rotation preprocessing is enabled
    rotation_enabled = config.get('rotation', {}).get('enabled', False)
    debug_visualization = config.get('rotation', {}).get('debug_visualization', False)
    
    # Update DPI to 600
    config['output']['plots']['dpi'] = 600
    
    # Create output directories
    create_output_directories(config)
    
    # Setup logging
    if config['output']['logs']['enabled']:
        log_file = os.path.join(
            config['output']['logs']['path'],
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['output']['logs']['filename']}"
        )
        setup_logging(log_file)
    else:
        setup_logging(None)  # Console-only logging
    
    logger = logging.getLogger(__name__)
    
    if rotation_enabled:
        if debug_visualization:
            logger.info("Starting visibility path planning with normal vector approach, building orientation preprocessing, and debug visualization enabled")
        else:
            logger.info("Starting visibility path planning with normal vector approach and building orientation preprocessing")
    else:
        logger.info("Starting visibility path planning with normal vector approach")
    
    log_memory_usage(logger, "Initial memory usage")
    
    # Record start time
    start_time = time.time()
    step_start_time = start_time
    
    try:
        # Load geometry data
        logger.info("Loading geometry data")
        if rotation_enabled:
            logger.info("Building orientation preprocessing is enabled")
            if debug_visualization:
                logger.info("Debug visualization is enabled")
        else:
            logger.info("Building orientation preprocessing is disabled")
            
        log_memory_usage(logger, "Before geometry loading")
        geometry_loader = GeometryLoader(config)
        building, obstacles, polygons = geometry_loader.load_geometries()
        
        # Get rotation parameters if enabled
        if rotation_enabled:
            rotation_angle, rotation_center, _ = geometry_loader.get_rotation_params()
            logger.info(f"Building rotation: {rotation_angle:.2f} degrees around {rotation_center}")
            
            # Print rotation debug information if available
            if debug_visualization:
                debug_info = geometry_loader.get_debug_info()
                print_rotation_debug_info(debug_info)
        
        log_memory_usage(logger, "After geometry loading")
        times['Geometry Loading'] = time.time() - step_start_time
        
        # Build the graph
        step_start_time = time.time()
        logger.info("Building the graph")
        log_memory_usage(logger, "Before graph building")
        graph_builder = GraphBuilder(config)
        G, grid_points = graph_builder.build_graph(building, obstacles)
        log_memory_usage(logger, "After graph building")
        times['Graph Building'] = time.time() - step_start_time
        
        # Calculate visibility using normal vector approach
        step_start_time = time.time()
        logger.info("Analyzing visibility with normal vector approach")
        log_memory_usage(logger, "Before visibility analysis")
        visibility_analyzer = VisibilityAnalyzer(config)
        segments, segment_visibility, edge_visibility, vrf = visibility_analyzer.analyze(
            G, grid_points, building, obstacles
        )
        log_memory_usage(logger, "After visibility analysis")
        times['Visibility Analysis'] = time.time() - step_start_time
        
        # Print time spent on visibility analysis
        print(f"Time spent on visibility analysis: {times['Visibility Analysis']:.2f} seconds")
        
        # Initialize visualizer for potential interruptions
        visualizer = PathVisualizer(config)
        
        # Run optimization
        step_start_time = time.time()
        logger.info("Running path optimization")
        log_memory_usage(logger, "Before optimization")
        
        # Memory optimization: Force garbage collection before optimization
        logger.info("Running garbage collection before optimization")
        gc.collect()
        
        optimizer = PathOptimizer(config)
        model, selected_edges = optimizer.optimize(
            G, segments, segment_visibility, edge_visibility, vrf
        )
        log_memory_usage(logger, "After optimization")
        times['Optimization'] = time.time() - step_start_time
        
        # Calculate total time
        total_time = time.time() - start_time
        times['Total'] = total_time
        
        # Calculate path metrics
        path_metrics = calculate_path_metrics(G, selected_edges, segments, segment_visibility)
        
        # Print metrics to terminal
        print("\n--- RESULTS ---")
        print(f"Path Length: {path_metrics['path_length']:.2f} units")
        print(f"Number of Edges: {path_metrics['num_edges']}")
        print(f"Visibility Ratio: {path_metrics['vrf']:.4f}")
        print(f"Optimization Time: {times['Optimization']:.2f} seconds")
        print(f"Total Processing Time: {times['Total']:.2f} seconds")
        
        # Check if we hit the time limit
        if model.status == 9:  # GRB.TIME_LIMIT
            if hasattr(model, 'MIPGap'):
                gap = model.MIPGap * 100  # Convert to percentage
                print(f"Reached time limit with {gap:.2f}% optimality gap")
        
        # Save path to file
        if rotation_enabled:
            save_path_to_file(G, selected_edges, rotation_angle, rotation_center, rotation_enabled)
        else:
            save_path_to_file(G, selected_edges, None, None, rotation_enabled)
        
        # Create visualization
        logger.info("Creating visualization")
        if rotation_enabled:
            visualizer.plot(
                G, building, obstacles, segments, selected_edges, path_metrics,
                rotation_angle, rotation_center
            )
        else:
            visualizer.plot(
                G, building, obstacles, segments, selected_edges, path_metrics
            )
        
        logger.info("Process completed successfully")
        
    except KeyboardInterrupt:
        # This should be caught by the signal handler
        pass
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        
        # Try to visualize what we have so far
        if G is not None and visualizer is None:
            visualizer = PathVisualizer(config)
        
        if visualizer is not None and G is not None and building is not None and obstacles is not None and segments is not None:
            path_metrics = calculate_path_metrics(G, selected_edges, segments, segment_visibility) if segments and segment_visibility else None
            
            if rotation_enabled:
                visualizer.plot(G, building, obstacles, segments, selected_edges, path_metrics,
                               rotation_angle, rotation_center)
            else:
                visualizer.plot(G, building, obstacles, segments, selected_edges, path_metrics)
        
        raise
    finally:
        # Dispose Gurobi model if it exists
        if model is not None:
            try:
                model.dispose()
                logger.info("Disposed Gurobi optimization model")
            except Exception as e:
                logger.error(f"Error disposing Gurobi model: {e}")
        
        # Clean up resources
        cleanup()

if __name__ == "__main__":
    main()