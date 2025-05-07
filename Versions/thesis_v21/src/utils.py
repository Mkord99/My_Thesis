"""
Utility functions for the visibility path planning application.
"""
import logging
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon

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

def log_memory_usage(logger, message):
    """Log current memory usage with a custom message."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    logger.info(f"{message} - Memory usage: {mem:.2f} MB")

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors.
    
    Args:
        vec1: First vector as a tuple (x, y)
        vec2: Second vector as a tuple (x, y)
        
    Returns:
        Angle in degrees
    """
    # Handle zero vectors
    if (vec1[0] == 0 and vec1[1] == 0) or (vec2[0] == 0 and vec2[1] == 0):
        return 0
        
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Handle potential numerical issues
    if mag1 * mag2 == 0:
        return 0
        
    cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def calculate_normal_vector(segment_vec, building, segment_midpoint):
    """
    Calculate the normal vector pointing outward from a building segment.
    
    Args:
        segment_vec: Segment vector as a tuple (x, y)
        building: MultiPolygon representing the building
        segment_midpoint: Point at the middle of the segment
        
    Returns:
        Normal vector as a tuple (x, y)
    """
    # Calculate both potential normal vectors (perpendicular to segment vector)
    normal1 = (-segment_vec[1], segment_vec[0])
    normal2 = (segment_vec[1], -segment_vec[0])
    
    # Normalize the vectors
    length1 = np.sqrt(normal1[0]**2 + normal1[1]**2)
    length2 = np.sqrt(normal2[0]**2 + normal2[1]**2)
    
    if length1 > 0:
        normal1 = (normal1[0]/length1, normal1[1]/length1)
    
    if length2 > 0:
        normal2 = (normal2[0]/length2, normal2[1]/length2)
    
    # Create test points a small distance along each normal vector
    test_dist = 0.1  # Small testing distance
    test_point1 = Point(segment_midpoint.x + normal1[0] * test_dist, 
                        segment_midpoint.y + normal1[1] * test_dist)
    test_point2 = Point(segment_midpoint.x + normal2[0] * test_dist, 
                        segment_midpoint.y + normal2[1] * test_dist)
    
    # Check which test point is outside the building (not contained by it)
    # That's the direction we want the normal vector to point
    if not building.contains(test_point1):
        return normal1
    elif not building.contains(test_point2):
        return normal2
    
    # If both seem to be inside (unlikely but possible with complex geometries),
    # return the first normal as a fallback
    return normal1

def is_within_angle_constraint(normal_vec, to_point_vec, max_angle):
    """
    Check if the angle between normal vector and vector to point is within constraints.
    
    Args:
        normal_vec: Normal vector as a tuple (x, y)
        to_point_vec: Vector to the point as a tuple (x, y)
        max_angle: Maximum allowed angle in degrees
        
    Returns:
        Boolean indicating if angle is within constraints
    """
    angle = calculate_angle(normal_vec, to_point_vec)
    return angle <= max_angle

def check_visibility_normal(point, segment_start, segment_end, normal_vec, building, obstacles, config):
    """
    Check if a point has visibility to a segment using normal vector approach.
    
    Args:
        point: Point object
        segment_start: Start point of segment
        segment_end: End point of segment
        normal_vec: Normal vector pointing outward from segment
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
    max_normal_angle = vis_config['max_normal_angle']
    
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
    if obstacles['visibility'].intersects(line_start) and obstacles['visibility'].intersects(line_end):
        return False
    
    # Calculate segment midpoint
    segment_midpoint_x = (segment_start.x + segment_end.x) / 2
    segment_midpoint_y = (segment_start.y + segment_end.y) / 2
    
    # Calculate vector from segment midpoint to point
    to_point_vec = (point.x - segment_midpoint_x, point.y - segment_midpoint_y)
    
    # Check if angle is within constraint
    if not is_within_angle_constraint(normal_vec, to_point_vec, max_normal_angle):
        return False
    
    return True

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

def save_visibility_data(filename, data, is_edge_visibility=False):
    """
    Save visibility data to a CSV file.
    
    Args:
        filename: Output filename
        data: Visibility data to save
        is_edge_visibility: Flag indicating if this is edge visibility data
    """
    import os
    import csv
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV file
    with open(filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        
        if is_edge_visibility:
            # For edge visibility data (edges -> segments)
            # Format: edge_start_node, edge_end_node, segment_idx1, segment_idx2, ...
            csv_writer.writerow(['edge_start_node', 'edge_end_node', 'visible_segments'])
            for edge, segments in data.items():
                if isinstance(edge, tuple) and len(edge) == 2:
                    # Format edge as "start_node,end_node"
                    row = [edge[0], edge[1]]
                    # Add all visible segments
                    if segments:
                        # Convert segment indices to strings and join with semicolons
                        segments_str = ';'.join(map(str, segments))
                        row.append(segments_str)
                    else:
                        row.append('')
                    csv_writer.writerow(row)
        else:
            # For segment visibility data (segments -> edges)
            # Format: segment_idx, edge_start_node1,edge_end_node1, edge_start_node2,edge_end_node2, ...
            csv_writer.writerow(['segment_idx', 'visible_edges'])
            for segment_idx, edges in data.items():
                row = [segment_idx]
                if edges:
                    # Convert edges to "start_node,end_node" format and join with semicolons
                    edges_str = ';'.join([f"{e[0]},{e[1]}" for e in edges])
                    row.append(edges_str)
                else:
                    row.append('')
                csv_writer.writerow(row)