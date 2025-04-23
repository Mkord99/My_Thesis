#!/usr/bin/env python3
"""
Main entry point for the visibility path planning application.
"""
import os
import json
import time
import logging
from datetime import datetime

from src.data_handler import GeometryLoader
from src.graph_builder import GraphBuilder
from src.visibility_analyzer import VisibilityAnalyzer
from src.optimizer import PathOptimizer
from src.visualizer import PathVisualizer
from src.utils import setup_logging

def main():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Create output directories if they don't exist
    os.makedirs(config['output']['logs']['path'], exist_ok=True)
    os.makedirs(config['output']['plots']['path'], exist_ok=True)
    
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
    logger.info("Starting visibility path planning")
    
    # Record start time
    start_time = time.time()
    
    # Load geometry data
    logger.info("Loading geometry data")
    geometry_loader = GeometryLoader(config)
    building, obstacles, polygons = geometry_loader.load_geometries()
    
    # Build the graph
    logger.info("Building the graph")
    graph_builder = GraphBuilder(config)
    G, grid_points = graph_builder.build_graph(building, obstacles)
    
    # Calculate visibility
    logger.info("Analyzing visibility")
    visibility_analyzer = VisibilityAnalyzer(config)
    segments, segment_visibility, edge_visibility, vrf = visibility_analyzer.analyze(
        G, grid_points, building, obstacles
    )
    
    # Run optimization
    logger.info("Running path optimization")
    optimizer = PathOptimizer(config)
    model, selected_edges = optimizer.optimize(
        G, segments, segment_visibility, edge_visibility, vrf
    )
    
    # Calculate optimization time
    optimization_time = time.time() - start_time
    logger.info(f"Total processing time: {optimization_time:.2f} seconds")
    
    # Create visualization
    logger.info("Creating visualization")
    visualizer = PathVisualizer(config)
    visualizer.plot(
        G, building, obstacles, segments, selected_edges
    )
    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()