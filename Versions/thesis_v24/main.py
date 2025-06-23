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
    
    # Create path nodes directory
    path_nodes_dir = os.path.join("output", "path_nodes")
    os.makedirs(path_nodes_dir, exist_ok=True)
    
    # Create orientation data directory if rotation is enabled
    if config.get('rotation', {}).get('enabled', False):
        orientation_dir = os.path.join("output", "orientation")
        os.makedirs(orientation_dir, exist_ok=True)
    
    # Create thesis plots directory if enabled
    if config['output'].get('thesis_plots', {}).get('enabled', False):
        thesis_plots_dir = config['output']['thesis_plots']['path']
        os.makedirs(thesis_plots_dir, exist_ok=True)
    
    # Create visibility heatmaps directory if enabled
    if config['output'].get('visibility_heatmaps', {}).get('enabled', False):
        heatmaps_dir = config['output']['visibility_heatmaps']['path']
        os.makedirs(heatmaps_dir, exist_ok=True)

def reconstruct_tour_from_edges(selected_edges):
    """
    Reconstruct the tour by following the directed edges in sequence.
    Simple approach: pick a start node and follow the edge directions.
    
    Args:
        selected_edges: List of selected edges (i, j) - directed edges
        
    Returns:
        List of nodes in correct tour order forming a closed loop
    """
    if not selected_edges:
        return []
    
    logger = logging.getLogger(__name__)
    logger.info(f"Reconstructing tour from {len(selected_edges)} directed edges")
    
    # Build adjacency dictionary: node -> [list of next nodes]
    adjacency = {}
    all_nodes = set()
    
    for i, j in selected_edges:
        all_nodes.add(i)
        all_nodes.add(j)
        
        if i not in adjacency:
            adjacency[i] = []
        adjacency[i].append(j)
    
    logger.info(f"Graph has {len(all_nodes)} nodes")
    
    # Pick any node from the selected edges as starting point
    start_node = selected_edges[0][0]  # Start from first edge's source node
    logger.info(f"Starting tour from node {start_node}")
    
    # Follow the directed edges to build the path
    tour = []
    current_node = start_node
    visited_edges = set()
    
    while True:
        tour.append(current_node)
        
        # Find the next node by following an unused outgoing edge
        next_node = None
        edge_to_use = None
        
        if current_node in adjacency:
            for candidate_next in adjacency[current_node]:
                edge = (current_node, candidate_next)
                if edge in selected_edges and edge not in visited_edges:
                    next_node = candidate_next
                    edge_to_use = edge
                    break
        
        if next_node is None:
            # No more unused edges from current node
            break
        
        # Mark this edge as used and move to next node
        visited_edges.add(edge_to_use)
        current_node = next_node
        
        # Check if we've returned to start (closed the loop)
        if current_node == start_node and len(visited_edges) == len(selected_edges):
            tour.append(start_node)  # Close the loop
            break
    
    # Verify we used all edges
    if len(visited_edges) != len(selected_edges):
        logger.warning(f"Used {len(visited_edges)} edges out of {len(selected_edges)} total edges")
        logger.warning("Path may not form a single connected tour")
    else:
        logger.info("Successfully used all edges in sequence")
    
    # Ensure the tour is closed
    if len(tour) > 1 and tour[0] != tour[-1]:
        tour.append(tour[0])
        logger.info("Manually closed the tour by adding starting node at the end")
    
    logger.info(f"Final tour: {len(tour)} nodes, visits {len(set(tour))} unique nodes")
    logger.info(f"Tour is closed: {tour[0] == tour[-1] if len(tour) > 1 else False}")
    
    return tour

def save_path_to_file(G, selected_edges):
    """
    Save the path to files as a proper tour sequence in multiple formats.
    
    Args:
        G: networkx DiGraph
        selected_edges: List of selected edges
    """
    logger = logging.getLogger(__name__)
    
    if not selected_edges:
        logger.warning("No path to save")
        return
    
    # Create the output directories if they don't exist
    path_dir = os.path.join("output", "path")
    path_nodes_dir = os.path.join("output", "path_nodes")
    os.makedirs(path_dir, exist_ok=True)
    os.makedirs(path_nodes_dir, exist_ok=True)
    
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
    
    # Save to path_nodes directory (NEW - for GIS software)
    # Format 1: Simple coordinate pairs (CSV-like for GIS)
    gis_coords_file = os.path.join(path_nodes_dir, "path_coordinates.txt")
    with open(gis_coords_file, 'w') as f:
        f.write("# Path coordinates for GIS software\n")
        f.write("# Format: X,Y (one coordinate pair per line)\n")
        f.write("# This forms a closed tour - first and last coordinates are the same\n")
        for x, y in path_coords:
            f.write(f"{x:.6f},{y:.6f}\n")
    
    # Format 2: CSV with headers (for easy import into GIS)
    csv_file = os.path.join(path_nodes_dir, "path_coordinates.csv")
    with open(csv_file, 'w') as f:
        f.write("X,Y,Point_ID\n")
        for i, (x, y) in enumerate(path_coords):
            f.write(f"{x:.6f},{y:.6f},{i+1}\n")
    
    # Format 3: Node sequence with coordinates
    node_coords_file = os.path.join(path_nodes_dir, "node_sequence_with_coordinates.txt")
    with open(node_coords_file, 'w') as f:
        f.write("# Node sequence with coordinates\n")
        f.write("# Format: NodeID X Y\n")
        f.write("# Closed tour: first and last nodes are the same\n")
        for i, node in enumerate(tour_nodes):
            pos = G.nodes[node]['pos']
            f.write(f"{node} {pos[0]:.6f} {pos[1]:.6f}\n")
    
    # Format 4: WKT (Well-Known Text) format for GIS
    wkt_file = os.path.join(path_nodes_dir, "path_linestring.wkt")
    with open(wkt_file, 'w') as f:
        f.write("# Path as WKT LINESTRING for GIS software\n")
        coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in path_coords])
        f.write(f"LINESTRING({coord_str})\n")
    
    # Original format in path directory (keep for compatibility)
    with open(os.path.join(path_dir, "path.txt"), 'w') as f:
        f.write("# Path coordinates in tour order (closed tour)\n")
        f.write("# Format: x, y\n")
        f.write("# Tour starts and ends at the same node\n")
        for x, y in path_coords:
            f.write(f"{x:.4f}, {y:.4f}\n")
    
    # Save tour node sequence
    with open(os.path.join(path_dir, "tour_nodes.txt"), 'w') as f:
        f.write("# Tour node sequence\n")
        f.write("# Node indices in tour order\n")
        for node in tour_nodes:
            f.write(f"{node}\n")
    
    # Create summary file
    summary_file = os.path.join(path_nodes_dir, "path_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("# Path Summary\n")
        f.write(f"Total nodes in tour: {len(tour_nodes)}\n")
        f.write(f"Unique nodes visited: {len(set(tour_nodes))}\n")
        f.write(f"Total edges used: {len(selected_edges)}\n")
        f.write(f"Tour is closed: {'Yes' if tour_nodes[0] == tour_nodes[-1] else 'No'}\n")
        f.write(f"Starting/ending node: {tour_nodes[0]}\n")
        
        # Calculate total path length
        total_length = 0
        for i in range(len(path_coords) - 1):
            x1, y1 = path_coords[i]
            x2, y2 = path_coords[i + 1]
            total_length += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        f.write(f"Total path length: {total_length:.2f} units\n")
        
        f.write("\n# Available files:\n")
        f.write("# - path_coordinates.txt: Simple X,Y format\n")
        f.write("# - path_coordinates.csv: CSV format with headers\n")
        f.write("# - node_sequence_with_coordinates.txt: Node IDs with coordinates\n")
        f.write("# - path_linestring.wkt: WKT format for GIS import\n")
    
    logger.info(f"Saved path in multiple formats to {path_nodes_dir}/")
    logger.info(f"Tour with {len(path_coords)} points, visiting {len(set(tour_nodes))} unique nodes")
    logger.info(f"Tour is properly closed: {tour_nodes[0] == tour_nodes[-1]}")
    
    # Validate tour connectivity
    if len(tour_nodes) > 1:
        missing_edges = []
        for i in range(len(tour_nodes) - 1):
            edge = (tour_nodes[i], tour_nodes[i + 1])
            if edge not in selected_edges:
                missing_edges.append(edge)
        
        if missing_edges:
            logger.warning(f"Tour contains {len(missing_edges)} edges not in selected_edges")
        else:
            logger.info("Tour successfully uses all selected edges in correct sequence")

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

def create_thesis_plots(graph_builder, building, obstacles, segments, rotation_params):
    """
    Create thesis-specific plots for analysis.
    
    Args:
        graph_builder: GraphBuilder instance
        building: MultiPolygon representing the building
        obstacles: Dictionary containing MultiPolygons for different obstacle types
        segments: List of segments
        rotation_params: Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle)
    """
    # Check if thesis plots are enabled
    if not config['output'].get('thesis_plots', {}).get('enabled', False):
        logger = logging.getLogger(__name__)
        logger.info("Thesis plots are disabled in config")
        return
    
    logger = logging.getLogger(__name__)
    logger.info("Creating thesis plots")
    
    # Build original graph (non-rotated) for plot 3
    G_original = graph_builder.build_original_graph(building, obstacles)
    
    # Build rotated graph for plot 4 (if rotation is enabled)
    G_rotated = None
    if rotation_enabled and rotation_params:
        rotation_angle, rotation_center, longest_edge_angle, target_angle = rotation_params
        G_rotated = graph_builder.build_rotated_graph(building, obstacles, rotation_angle, rotation_center)
    
    # Create thesis plots using the visualizer
    visualizer.create_thesis_plots(G_original, building, obstacles, segments, G_rotated, rotation_params)

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
        
        # Create thesis plots if enabled (before optimization)
        if config['output'].get('thesis_plots', {}).get('enabled', False):
            logger.info("Creating thesis plots")
            rotation_params_for_plots = None
            if rotation_enabled:
                rotation_params_for_plots = (rotation_angle, rotation_center, longest_edge_angle, target_angle)
            create_thesis_plots(graph_builder, building, obstacles, segments, rotation_params_for_plots)
        
        # Create visibility heatmaps if enabled (after visibility analysis)
        if config['output'].get('visibility_heatmaps', {}).get('enabled', False):
            logger.info("Creating visibility heatmaps (basic)")
            visualizer.create_visibility_heatmaps(G, building, obstacles, segments, edge_visibility, segment_visibility, vrf)
        
        # Run optimization only if enabled
        optimization_enabled = config['optimization'].get('enabled', True)
        if optimization_enabled:
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
            
            # Create additional heatmap with optimization results if heatmaps are enabled
            if config['output'].get('visibility_heatmaps', {}).get('enabled', False):
                logger.info("Creating VRF heatmap with optimized path overlay")
                visualizer.create_visibility_heatmaps(G, building, obstacles, segments, edge_visibility, segment_visibility, vrf, selected_edges)
            
            # Create visualization with optimization results
            logger.info("Creating visualization with optimization results")
            if rotation_enabled:
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, path_metrics,
                    rotation_params=(rotation_angle, rotation_center, longest_edge_angle, target_angle)
                )
            else:
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, path_metrics
                )
        else:
            # Skip optimization
            logger.info("Optimization is disabled in config - skipping optimization step")
            selected_edges = []
            
            # Calculate total time for preprocessing only
            total_time = time.time() - start_time
            times['Total'] = total_time
            
            # Print preprocessing completion message
            print("\n--- PREPROCESSING COMPLETED ---")
            print(f"Graph nodes: {G.number_of_nodes()}")
            print(f"Graph edges: {G.number_of_edges()}")
            print(f"Building segments: {len(segments)}")
            print(f"Total preprocessing time: {times['Total']:.2f} seconds")
            print("Optimization was skipped (disabled in config)")
            
            # Create visualization without optimization results
            logger.info("Creating visualization without optimization results")
            if rotation_enabled:
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, None,
                    rotation_params=(rotation_angle, rotation_center, longest_edge_angle, target_angle)
                )
            else:
                visualizer.plot(
                    G, building, obstacles, segments, selected_edges, None
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