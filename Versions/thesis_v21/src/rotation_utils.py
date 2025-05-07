"""
Rotation utilities for building orientation preprocessing and result postprocessing.
With enhanced debugging and verification capabilities.
"""
import logging
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

def find_longest_edge_and_angle(building, debug=False):
    """
    Find the longest edge in the building MultiPolygon and its angle with north.
    
    Args:
        building: Building MultiPolygon
        debug: Whether to return debug information
        
    Returns:
        If debug=False: Tuple of (longest edge length, angle with north, start point, end point, polygon)
        If debug=True: Tuple of (longest edge length, angle with north, start point, end point, polygon, all_edges)
    """
    longest_edge_length = 0
    longest_edge_angle = 0
    longest_edge_start = None
    longest_edge_end = None
    longest_poly = None
    
    # Store all edges for debugging if needed
    all_edges = []
    
    # Analyze each polygon in the MultiPolygon
    for poly in building.geoms:
        # Get coordinates from the exterior boundary
        coords = list(poly.exterior.coords)
        
        # Check each edge
        for i in range(len(coords) - 1):  # Last point is same as first in closed polygons
            start_point = Point(coords[i])
            end_point = Point(coords[i+1])
            
            # Calculate edge length
            edge_length = start_point.distance(end_point)
            
            # Calculate angle with north for all edges when debugging
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            
            # Calculate angle with north (y-axis)
            angle_rad = np.arctan2(dx, dy)  # Returns angle in [-pi, pi]
            angle_deg = np.degrees(angle_rad)
            
            # Convert to positive angle in [0, 360)
            if angle_deg < 0:
                angle_deg += 360
            
            # Ensure angle is between 0-180 (if > 180, use opposite direction)
            if angle_deg > 180:
                angle_deg = angle_deg - 180
                # For debug data, swap start and end points to maintain consistent direction
                if debug:
                    temp_start = start_point
                    temp_end = end_point
                    start_point_for_debug = end_point
                    end_point_for_debug = temp_start
                else:
                    start_point_for_debug = start_point
                    end_point_for_debug = end_point
            else:
                start_point_for_debug = start_point
                end_point_for_debug = end_point
            
            # Store edge information for debugging
            if debug:
                all_edges.append({
                    'start': start_point_for_debug,
                    'end': end_point_for_debug,
                    'length': edge_length,
                    'angle': angle_deg,
                    'poly_index': id(poly) # Use object id as index
                })
            
            # If this is the longest edge so far, update information
            if edge_length > longest_edge_length:
                longest_edge_length = edge_length
                
                # Calculate directional vector of the edge
                dx = end_point.x - start_point.x
                dy = end_point.y - start_point.y
                
                # Calculate angle with north (y-axis)
                angle_rad = np.arctan2(dx, dy)  # Returns angle in [-pi, pi]
                angle_deg = np.degrees(angle_rad)
                
                # Convert to positive angle in [0, 360)
                if angle_deg < 0:
                    angle_deg += 360
                
                # Ensure angle is between 0-180 (if > 180, use opposite direction)
                if angle_deg > 180:
                    angle_deg = angle_deg - 180
                    # Swap start and end points to maintain correct edge direction
                    temp = start_point
                    start_point = end_point
                    end_point = temp
                
                longest_edge_angle = angle_deg
                longest_edge_start = start_point
                longest_edge_end = end_point
                longest_poly = poly
    
    if debug:
        return longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly, all_edges
    else:
        return longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly

def get_target_angle(longest_edge_angle):
    """
    Determine target angle based on the classification:
    If angle is between 0° and 45°, target is 0° (North)
    If angle is between 45° and 135°, target is 90° (East)
    If angle is between 135° and 180°, target is 180° (South)
    
    Args:
        longest_edge_angle: Current angle with north
        
    Returns:
        Target angle (0, 90, or 180)
    """
    if 0 <= longest_edge_angle < 45:
        return 0  # Align with North
    elif 45 <= longest_edge_angle < 135:
        return 90  # Align with East
    else:  # 135 <= longest_edge_angle <= 180
        return 180  # Align with South

def calculate_rotation_angle(longest_edge_angle, target_angle):
    """
    Calculate the rotation angle needed to align the edge with the target angle.
    Positive values will rotate counter-clockwise.
    Negative values will rotate clockwise.
    
    Args:
        longest_edge_angle: Current angle with north
        target_angle: Target angle for alignment
        
    Returns:
        Rotation angle in degrees
    """
    # The rotation needed is the difference between target and current angle
    rotation_angle = target_angle - longest_edge_angle
    
    # Normalize to range [-180, 180] for most efficient rotation
    while rotation_angle > 180:
        rotation_angle -= 360
    while rotation_angle < -180:
        rotation_angle += 360
        
    return rotation_angle

def rotate_point(point, angle_deg, origin=(0, 0)):
    """
    Rotate a point around an origin by a specified angle in degrees.
    Positive angle = counter-clockwise rotation, negative angle = clockwise rotation.
    
    Args:
        point: Tuple or Point object with coordinates (x, y)
        angle_deg: Rotation angle in degrees
        origin: Tuple with coordinates (x, y) of rotation origin
        
    Returns:
        Tuple of rotated coordinates (x, y)
    """
    # Convert angle to radians
    angle_rad = math.radians(angle_deg)
    
    # Get coordinates
    if isinstance(point, Point):
        x, y = point.x, point.y
    else:
        x, y = point
    
    ox, oy = origin
    
    # Translate point to origin
    px, py = x - ox, y - oy
    
    # Apply rotation matrix
    qx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
    qy = px * math.sin(angle_rad) + py * math.cos(angle_rad)
    
    # Translate back
    return (qx + ox, qy + oy)

def rotate_polygon(polygon, angle_deg, origin=(0, 0)):
    """
    Rotate a polygon around an origin by a specified angle in degrees.
    
    Args:
        polygon: Shapely Polygon
        angle_deg: Rotation angle in degrees
        origin: Tuple with coordinates (x, y) of rotation origin
        
    Returns:
        Rotated Shapely Polygon
    """
    # Get exterior coordinates
    coords = list(polygon.exterior.coords)
    
    # Rotate each coordinate
    rotated_coords = [rotate_point(p, angle_deg, origin) for p in coords]
    
    # Create a new polygon with rotated coordinates
    return Polygon(rotated_coords)

def rotate_multipolygon(multipolygon, angle_deg, origin=(0, 0)):
    """
    Rotate a multipolygon around an origin by a specified angle in degrees.
    
    Args:
        multipolygon: Shapely MultiPolygon
        angle_deg: Rotation angle in degrees
        origin: Tuple with coordinates (x, y) of rotation origin
        
    Returns:
        Rotated Shapely MultiPolygon
    """
    # Rotate each polygon in the multipolygon
    rotated_polygons = [rotate_polygon(poly, angle_deg, origin) for poly in multipolygon.geoms]
    
    # Create a new multipolygon with rotated polygons
    return MultiPolygon(rotated_polygons)

def get_rotation_center(building):
    """
    Calculate the center point of a building for rotation.
    
    Args:
        building: Building MultiPolygon
        
    Returns:
        Tuple of (center_x, center_y)
    """
    # Calculate the centroid of the MultiPolygon
    centroid = building.centroid
    return (centroid.x, centroid.y)

def visualize_building_edges(building, fig=None, ax=None, title=None):
    """
    Visualize a building with its edges highlighted, showing the longest edge.
    
    Args:
        building: Building MultiPolygon
        fig: Matplotlib figure (optional)
        ax: Matplotlib axis (optional)
        title: Plot title (optional)
        
    Returns:
        Tuple of (fig, ax, longest_edge_info)
    """
    # Get the longest edge and all edges for debugging
    longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly, all_edges = \
        find_longest_edge_and_angle(building, debug=True)
    
    # Sort edges by length (descending)
    all_edges.sort(key=lambda x: x['length'], reverse=True)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all building polygons
    for poly in building.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, 'g-', linewidth=1, alpha=0.7)
        ax.fill(x, y, alpha=0.2, color='green')
    
    # Plot all edges
    for i, edge in enumerate(all_edges):
        start = edge['start']
        end = edge['end']
        
        # Only plot top 10 longest edges with numbers
        if i < 10:
            ax.plot([start.x, end.x], [start.y, end.y], 'b-', linewidth=1+2*(1-i/10))
            # Add edge number and length
            midpoint_x = (start.x + end.x) / 2
            midpoint_y = (start.y + end.y) / 2
            ax.text(midpoint_x, midpoint_y, f"{i+1}: {edge['length']:.1f}m", 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Highlight the longest edge
    ax.plot([longest_edge_start.x, longest_edge_end.x], 
            [longest_edge_start.y, longest_edge_end.y], 
            'r-', linewidth=3, label=f'Longest Edge: {longest_edge_length:.2f}m')
    
    # Plot angle annotation
    midpoint_x = (longest_edge_start.x + longest_edge_end.x) / 2
    midpoint_y = (longest_edge_start.y + longest_edge_end.y) / 2
    
    # Draw a line representing North
    north_length = longest_edge_length * 0.3
    ax.plot([midpoint_x, midpoint_x], 
            [midpoint_y, midpoint_y + north_length], 
            'k--', linewidth=1, label='North')
    
    # Add angle annotation
    ax.annotate(f'Angle: {longest_edge_angle:.2f}°',
               xy=(midpoint_x, midpoint_y),
               xytext=(midpoint_x + longest_edge_length * 0.2, 
                       midpoint_y + longest_edge_length * 0.2),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
               fontsize=10, ha='center')
    
    # Add target angle
    target_angle = get_target_angle(longest_edge_angle)
    ax.set_title(f'{title or "Building Edges Analysis"}\n'
                f'Longest Edge Angle: {longest_edge_angle:.2f}°, '
                f'Target Angle: {target_angle}°, '
                f'Rotation Needed: {calculate_rotation_angle(longest_edge_angle, target_angle):.2f}°')
    
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    
    return fig, ax, {
        'longest_edge_length': longest_edge_length,
        'longest_edge_angle': longest_edge_angle,
        'target_angle': target_angle,
        'rotation_angle': calculate_rotation_angle(longest_edge_angle, target_angle)
    }

def process_building_orientation(building, debug_visualization=False):
    """
    Analyze building orientation and determine rotation parameters.
    
    Args:
        building: Building MultiPolygon
        debug_visualization: Whether to generate debug visualizations
        
    Returns:
        Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle, debug_info)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize debug info
    debug_info = {}
    
    # Find the longest edge and its angle
    if debug_visualization:
        longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly, all_edges = \
            find_longest_edge_and_angle(building, debug=True)
        debug_info['edges'] = all_edges
    else:
        longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly = \
            find_longest_edge_and_angle(building)
    
    logger.info(f"Longest edge length: {longest_edge_length:.2f}, angle with north: {longest_edge_angle:.2f} degrees")
    
    # Determine target angle based on classification
    target_angle = get_target_angle(longest_edge_angle)
    logger.info(f"Target angle: {target_angle} degrees")
    
    # Calculate rotation angle
    rotation_angle = calculate_rotation_angle(longest_edge_angle, target_angle)
    
    # For clarity in logs, show whether it's clockwise or counter-clockwise
    rotation_direction = "counter-clockwise" if rotation_angle > 0 else "clockwise"
    logger.info(f"Rotation angle needed: {rotation_angle:.2f} degrees ({rotation_direction})")
    
    # Calculate rotation center
    rotation_center = get_rotation_center(building)
    logger.info(f"Rotation center: {rotation_center}")
    
    # Create debug visualization if requested
    if debug_visualization:
        fig, ax, edge_info = visualize_building_edges(building, title="Before Rotation")
        debug_info['before_rotation_fig'] = fig
        debug_info['edge_info'] = edge_info
        
        # Add the visualization to debug info
        debug_info['longest_edge'] = {
            'start': (longest_edge_start.x, longest_edge_start.y),
            'end': (longest_edge_end.x, longest_edge_end.y),
            'length': longest_edge_length,
            'angle': longest_edge_angle
        }
    
    return rotation_angle, rotation_center, longest_edge_angle, target_angle, debug_info

def rotate_geometries(building, obstacles, rotation_angle, rotation_center, debug_visualization=False):
    """
    Rotate building and obstacles by the given angle around the center.
    
    Args:
        building: Building MultiPolygon
        obstacles: Dictionary containing MultiPolygons for different obstacle types
        rotation_angle: Rotation angle in degrees
        rotation_center: Tuple of (center_x, center_y)
        debug_visualization: Whether to generate debug visualizations
        
    Returns:
        Tuple of (rotated building, rotated obstacles, debug_info)
    """
    logger = logging.getLogger(__name__)
    debug_info = {}
    
    # Rotate building
    rotated_building = rotate_multipolygon(building, rotation_angle, rotation_center)
    
    # Rotate obstacles
    rotated_obstacles = {}
    for key, obstacle_multipolygon in obstacles.items():
        rotated_obstacles[key] = rotate_multipolygon(obstacle_multipolygon, rotation_angle, rotation_center)
    
    # Create verification visualization if requested
    if debug_visualization:
        # Analyze rotated building
        fig, ax, edge_info = visualize_building_edges(rotated_building, title="After Rotation")
        debug_info['after_rotation_fig'] = fig
        debug_info['rotated_edge_info'] = edge_info
        
        # Verify rotation success
        original_angle, rotated_angle, target = verify_rotation(building, rotated_building)
        logger.info(f"Verification - Original angle: {original_angle:.2f}°, "
                    f"Rotated angle: {rotated_angle:.2f}°, Target: {target}°")
        
        # Calculate error
        error = abs(rotated_angle - target)
        logger.info(f"Rotation error: {error:.4f}° from target")
        
        # Add verification info to debug data
        debug_info['verification'] = {
            'original_angle': original_angle,
            'rotated_angle': rotated_angle,
            'target_angle': target,
            'error': error
        }
    
    return rotated_building, rotated_obstacles, debug_info

def rotate_path(path, rotation_angle, rotation_center, inverse=False):
    """
    Rotate path coordinates by the given angle around the center.
    
    Args:
        path: List of (x, y) coordinates or Point objects
        rotation_angle: Rotation angle in degrees
        rotation_center: Tuple of (center_x, center_y)
        inverse: Boolean indicating if rotation should be inverted (for reverse rotation)
        
    Returns:
        List of rotated coordinates
    """
    # Invert rotation angle if needed
    if inverse:
        rotation_angle = -rotation_angle
    
    # Rotate each point in the path
    rotated_path = [rotate_point(p, rotation_angle, rotation_center) for p in path]
    
    return rotated_path

def verify_rotation(building, rotated_building):
    """
    Verify that the rotation was applied correctly by checking the angle
    of the longest edge in the rotated building.
    
    Args:
        building: Original MultiPolygon
        rotated_building: Rotated MultiPolygon
        
    Returns:
        Tuple of (original_angle, rotated_angle, target_angle)
    """
    # Get original angle and target
    _, original_angle, _, _, _ = find_longest_edge_and_angle(building)
    target_angle = get_target_angle(original_angle)
    
    # Get angle after rotation
    _, rotated_angle, _, _, _ = find_longest_edge_and_angle(rotated_building)
    
    return original_angle, rotated_angle, target_angle

def save_debug_visualizations(debug_info, output_dir="output/orientation"):
    """
    Save debug visualizations to files.
    
    Args:
        debug_info: Debug information dictionary
        output_dir: Output directory for visualization files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save before rotation figure if available
    if 'before_rotation_fig' in debug_info:
        before_path = os.path.join(output_dir, "before_rotation.png")
        debug_info['before_rotation_fig'].savefig(before_path, dpi=300, bbox_inches='tight')
        print(f"Saved before rotation visualization to {before_path}")
    
    # Save after rotation figure if available
    if 'after_rotation_fig' in debug_info:
        after_path = os.path.join(output_dir, "after_rotation.png")
        debug_info['after_rotation_fig'].savefig(after_path, dpi=300, bbox_inches='tight')
        print(f"Saved after rotation visualization to {after_path}")