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
            visualizer.plot(G, building, obstacles, segments, selected_edges, path_metrics)
    
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

def main():
    global G, grid_points, building, obstacles, segments, segment_visibility, edge_visibility, vrf, selected_edges, config, visualizer, model, times
    
    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
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
    logger.info("Starting visibility path planning with normal vector approach")
    log_memory_usage(logger, "Initial memory usage")
    
    # Record start time
    start_time = time.time()
    step_start_time = start_time
    
    try:
        # Load geometry data
        logger.info("Loading geometry data")
        log_memory_usage(logger, "Before geometry loading")
        geometry_loader = GeometryLoader(config)
        building, obstacles, polygons = geometry_loader.load_geometries()
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
        
        # Create visualization
        logger.info("Creating visualization")
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