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
longest_edge_angle = 0
target_angle = 0
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
                          rotation_params=(rotation_angle, rotation_center, longest_edge_angle, target_angle) if rotation_enabled else None)
    
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

def reconstruct_tour_from_edges(selected_edges):
    """
    Reconstruct the tour from selected edges by following the directed path.
    This creates a proper tour that starts at one node and follows the edges
    in the correct direction until returning to the starting node.
    
    Args:
        selected_edges: List of selected edges (i, j)
        
    Returns:
        List of nodes in tour order
    """
    if not selected_edges:
        return []
    
    # Build adjacency list for outgoing edges
    adjacency = {}
    edge_count = {}  # Track how many times each edge should be used
    
    for i, j in selected_edges:
        if i not in adjacency:
            adjacency[i] = []
        adjacency[i].append(j)
        
        # Count edge occurrences (for cases where an edge might be selected multiple times)
        edge = (i, j)
        edge_count[edge] = edge_count.get(edge, 0) + 1
    
    # Start from the first node of the first edge
    start_node = selected_edges[0][0]
    
    # Follow the path
    tour = []
    current_node = start_node
    used_edges = {}  # Track how many times each edge has been used
    
    # Initialize used_edges counter
    for edge in selected_edges:
        used_edges[edge] = 0
    
    while len([e for e, count in used_edges.items() if count < edge_count[e]]) > 0:
        tour.append(current_node)
        
        # Find next available edge from current node
        next_node = None
        selected_edge = None
        
        if current_node in adjacency:
            for candidate in adjacency[current_node]:
                edge = (current_node, candidate)
                if edge in used_edges and used_edges[edge] < edge_count[edge]:
                    next_node = candidate
                    selected_edge = edge
                    break
        
        if next_node is not None:
            used_edges[selected_edge] += 1
            current_node = next_node
        else:
            # No available edges from current node, tour might be incomplete
            break
    
    return tour

def save_path_to_file(G, selected_edges):
    """
    Save the path to a file as a proper tour sequence.
    
    Args:
        G: networkx DiGraph
        selected_edges: List of selected edges
    """
    logger = logging.getLogger(__name__)
    
    if not selected_edges:
        logger.warning("No path to save")
        return
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join("output", "path")
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconstruct the tour from selected edges
    tour_nodes = reconstruct_tour_from_edges(selected_edges)
    
    if not tour_nodes:
        logger.warning("Could not reconstruct tour from selected edges")
        return
    
    # Get coordinates for each node in the tour
    path_coords = []
    for node in tour_nodes:
        pos = G.nodes[node]['pos']
        path_coords.append((pos[0], pos[1]))
    
    # Close the tour by adding the first node at the end if it's not already closed
    if len(tour_nodes) > 1 and tour_nodes[-1] != tour_nodes[0]:
        pos = G.nodes[tour_nodes[0]]['pos']
        path_coords.append((pos[0], pos[1]))
        logger.info("Added starting node at the end to close the tour")
    
    # Save path coordinates
    with open(os.path.join(output_dir, "path.txt"), 'w') as f:
        f.write("# Path coordinates in tour order (closed tour)\n")
        f.write("# Format: x, y\n")
        f.write("# Tour starts and ends at the same node\n")
        for i, (x, y) in enumerate(path_coords):
            f.write(f"{x:.4f}, {y:.4f}\n")
    
    # Also save tour node sequence
    with open(os.path.join(output_dir, "tour_nodes.txt"), 'w') as f:
        f.write("# Tour node sequence\n")
        f.write("# Node indices in tour order\n")
        for i, node in enumerate(tour_nodes):
            f.write(f"{node}\n")
        # Add starting node at the end if not already there
        if len(tour_nodes) > 1 and tour_nodes[-1] != tour_nodes[0]:
            f.write(f"{tour_nodes[0]}\n")
    
    logger.info(f"Saved tour path with {len(path_coords)} points to {output_dir}/path.txt")
    logger.info(f"Tour visits {len(set(tour_nodes))} unique nodes with {len(tour_nodes)} total steps")
    
    # Log some tour statistics
    unique_nodes = len(set(tour_nodes))
    total_steps = len(tour_nodes)
    if unique_nodes < total_steps:
        logger.info(f"Tour includes {total_steps - unique_nodes} revisited nodes (tie points)")

def print_orientation_info(rotation_angle, rotation_center, longest_edge_angle, target_angle):
    """
    Print building orientation information to console.
    
    Args:
        rotation_angle: Rotation angle
        rotation_center: Rotation center
        longest_edge_angle: Angle of the longest edge with north
        target_angle: Target alignment angle
    """
    print("\n--- BUILDING ORIENTATION INFORMATION ---")
    print(f"Longest edge angle with north: {longest_edge_angle:.2f}°")
    print(f"Target alignment angle: {target_angle}°")
    print(f"Grid alignment rotation: {rotation_angle:.2f}° (applied to grid)")
    print(f"Rotation center: ({rotation_center[0]:.2f}, {rotation_center[1]:.2f})")
    
    # Determine orientation classification
    if target_angle == 0:
        orientation = "Vertical (North)"
    elif target_angle == 90:
        orientation = "Horizontal (East)"
    else:  # target_angle == 180
        orientation = "Vertical (South)"
    
    print(f"Grid aligned with longest edge in: {orientation} orientation")
    print()

def main():
    global G, grid_points, building, obstacles, segments, segment_visibility, edge_visibility 
    global vrf, selected_edges, config, visualizer, model, times
    global rotation_angle, rotation_center, longest_edge_angle, target_angle, rotation_enabled
    global geometry_loader
    
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
            logger.info("Starting visibility path planning with grid aligned to building orientation (debug enabled)")
        else:
            logger.info("Starting visibility path planning with grid aligned to building orientation")
    else:
        logger.info("Starting visibility path planning with normal vector approach")
    
    log_memory_usage(logger, "Initial memory usage")
    
    # Record start time
    start_time = time.time()
    step_start_time = start_time
    
    try:
        # Load geometry data (unrotated)
        logger.info("Loading geometry data")
        if rotation_enabled:
            logger.info("Building orientation analysis is enabled")
            if debug_visualization:
                logger.info("Debug visualization is enabled")
        else:
            logger.info("Building orientation analysis is disabled")
            
        log_memory_usage(logger, "Before geometry loading")
        geometry_loader = GeometryLoader(config)
        building, obstacles, polygons = geometry_loader.load_geometries()
        
        # Get rotation parameters if enabled
        if rotation_enabled:
            rotation_params = geometry_loader.get_rotation_params()
            rotation_angle, rotation_center, longest_edge_angle, target_angle, _ = rotation_params
            
            # Print orientation information
            print_orientation_info(rotation_angle, rotation_center, longest_edge_angle, target_angle)
        
        log_memory_usage(logger, "After geometry loading")
        times['Geometry Loading'] = time.time() - step_start_time
        
        # Build the graph with alignment to the building's orientation
        step_start_time = time.time()
        logger.info("Building the graph")
        log_memory_usage(logger, "Before graph building")
        graph_builder = GraphBuilder(config)
        
        if rotation_enabled:
            # Build graph with alignment to the building's orientation
            G, grid_points = graph_builder.build_graph(
                building, obstacles, rotation_angle, rotation_center
            )
        else:
            # Build standard grid
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
        
        # Save path to file (with proper tour reconstruction)
        save_path_to_file(G, selected_edges)
        
        # Create visualization
        logger.info("Creating visualization")
        if rotation_enabled:
            visualizer.plot(
                G, building, obstacles, segments, selected_edges, path_metrics,
                rotation_params=(rotation_angle, rotation_center, longest_edge_angle, target_angle)
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
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, path_metrics,
                    rotation_params=(rotation_angle, rotation_center, longest_edge_angle, target_angle)
                )
            else:
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, path_metrics
                )
        
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