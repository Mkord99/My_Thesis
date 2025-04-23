"""
Utility functions for the visibility path planning application.
"""
import logging
import numpy as np
import math
from shapely.geometry import LineString, Point

def setup_logging(log_file=None):
    """Configure logging to file and console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors.
    
    Args:
        vec1: First vector as a tuple (x, y)
        vec2: Second vector as a tuple (x, y)
        
    Returns:
        Angle in degrees
    """
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Handle potential numerical issues
    cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def check_visibility(point, segment_start, segment_end, building, obstacles, config):
    """
    Check if a point has visibility to a segment based on constraints.
    
    Args:
        point: Point object
        segment_start: Start point of segment
        segment_end: End point of segment
        building: Building polygon
        obstacles: Obstacle polygons
        config: Configuration dictionary
        
    Returns:
        Boolean indicating visibility
    """
    # Extract visibility constraints from config
    vis_config = config['visibility']['visibility_constraints']
    min_distance = vis_config['min_distance']
    max_distance = vis_config['max_distance']
    min_angle = vis_config['min_angle']
    max_angle = vis_config['max_angle']
    
    # Calculate distances to segment endpoints
    d_start = point.distance(segment_start)
    d_end = point.distance(segment_end)
    
    # Check if distances are within range
    if not (min_distance <= d_start <= max_distance and 
            min_distance <= d_end <= max_distance):
        return False
    
    # Create sight lines to segment endpoints
    line_start = LineString([point, segment_start])
    line_end = LineString([point, segment_end])
    
    # Check if sight lines touch the building (required for proper visibility)
    touches_start = line_start.touches(building)
    touches_end = line_end.touches(building)
    
    if not (touches_start and touches_end):
        return False
    
    # Check if visibility is blocked by obstacles
    for obstacle in obstacles.geoms:
        if obstacle.intersects(line_start) and obstacle.intersects(line_end):
            return False
    
    # Calculate angle between point-segment vector and segment vector
    segment_vec = (segment_end.x - segment_start.x, segment_end.y - segment_start.y)
    point_to_start_vec = (segment_start.x - point.x, segment_start.y - point.y)
    point_to_end_vec = (segment_end.x - point.x, segment_end.y - point.y)
    
    angle_start = calculate_angle(point_to_start_vec, segment_vec)
    angle_end = calculate_angle(point_to_end_vec, segment_vec)
    
    # Check if angles are within range
    if not (min_angle <= angle_start <= max_angle and 
            min_angle <= angle_end <= max_angle):
        return False
    
    return True

def get_subtour(nodes, selected_edges):
    """
    Identify a subtour in the graph.
    
    Args:
        nodes: List of graph nodes
        selected_edges: List of selected edges
        
    Returns:
        A subtour component or None if no subtour exists
    """
    import networkx as nx
    
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(selected_edges)
    
    components = list(nx.connected_components(H))
    
    # One component = no subtour
    if len(components) == 1:
        return None
    
    # Return the smallest component as a subtour
    return min(components, key=len)