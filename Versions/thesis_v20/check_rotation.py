import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
import math

def find_longest_edge_and_angle(building):
    """Find the longest edge in the building MultiPolygon and its angle with north."""
    longest_edge_length = 0
    longest_edge_angle = 0
    longest_edge_start = None
    longest_edge_end = None
    longest_poly = None
    
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
    
    return longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly

def get_target_angle(longest_edge_angle):
    """
    Determine target angle based on the new classification:
    If angle is between 0° and 45°, target is 0° (North)
    If angle is between 45° and 135°, target is 90° (East)
    If angle is between 135° and 180°, target is 180° (South)
    """
    if 0 <= longest_edge_angle < 45:
        return 0  # Align with North
    elif 45 <= longest_edge_angle < 135:
        return 90  # Align with East
    else:  # 135 <= longest_edge_angle <= 180
        return 180  # Align with South

def calculate_rotation_angle(longest_edge_angle, target_angle):
    """Calculate the rotation angle needed."""
    # Calculate the difference between current and target angles
    rotation_angle = target_angle - longest_edge_angle
    
    # Normalize to range [-180, 180] for most efficient rotation
    while rotation_angle > 180:
        rotation_angle -= 360
    while rotation_angle < -180:
        rotation_angle += 360
        
    return rotation_angle

def visualize_building_and_longest_edge(building, longest_edge_start, longest_edge_end, longest_edge_angle, longest_poly):
    """Create a visualization of the building with the longest edge highlighted."""
    plt.figure(figsize=(12, 10))
    
    # Plot all buildings
    for poly in building.geoms:
        x, y = poly.exterior.xy
        if poly == longest_poly:
            plt.plot(x, y, 'g-', linewidth=2, alpha=0.7)
            plt.fill(x, y, alpha=0.2, color='green')
        else:
            plt.plot(x, y, 'b-', linewidth=1, alpha=0.5)
            plt.fill(x, y, alpha=0.1, color='blue')
    
    # Highlight longest edge
    plt.plot([longest_edge_start.x, longest_edge_end.x], 
             [longest_edge_start.y, longest_edge_end.y], 
             'r-', linewidth=3, label=f"Longest Edge: {longest_edge_length:.2f} units")
    
    # Add edge midpoint marker
    midpoint_x = (longest_edge_start.x + longest_edge_end.x) / 2
    midpoint_y = (longest_edge_start.y + longest_edge_end.y) / 2
    plt.plot(midpoint_x, midpoint_y, 'ro', markersize=8)
    
    # Add angle visualization
    edge_vector = (longest_edge_end.x - longest_edge_start.x, longest_edge_end.y - longest_edge_start.y)
    edge_length = np.sqrt(edge_vector[0]**2 + edge_vector[1]**2)
    normalized_edge = (edge_vector[0]/edge_length, edge_vector[1]/edge_length)
    
    # Draw north direction from midpoint
    north_length = edge_length * 0.3  # 30% of edge length
    plt.arrow(midpoint_x, midpoint_y, 0, north_length, head_width=north_length*0.1, 
              head_length=north_length*0.1, fc='blue', ec='blue', label="North")
    plt.text(midpoint_x, midpoint_y + north_length*1.1, "N", ha='center', fontsize=12)
    
    # Draw south direction from midpoint
    plt.arrow(midpoint_x, midpoint_y, 0, -north_length, head_width=north_length*0.1, 
              head_length=north_length*0.1, fc='purple', ec='purple', label="South")
    plt.text(midpoint_x, midpoint_y - north_length*1.1, "S", ha='center', fontsize=12)
    
    # Draw edge direction from midpoint
    plt.arrow(midpoint_x, midpoint_y, 
              normalized_edge[0] * north_length, normalized_edge[1] * north_length, 
              head_width=north_length*0.1, head_length=north_length*0.1, 
              fc='red', ec='red')
    
    # Draw arc showing the angle
    radius = north_length * 0.5
    angles = np.linspace(0, longest_edge_angle, 100)
    arc_x = midpoint_x + radius * np.sin(np.radians(angles))
    arc_y = midpoint_y + radius * np.cos(np.radians(angles))
    plt.plot(arc_x, arc_y, 'k-', linewidth=1.5)
    
    # Add angle text
    angle_text_radius = radius * 1.2
    angle_text_angle = longest_edge_angle / 2  # Halfway through the arc
    angle_text_x = midpoint_x + angle_text_radius * np.sin(np.radians(angle_text_angle))
    angle_text_y = midpoint_y + angle_text_radius * np.cos(np.radians(angle_text_angle))
    plt.text(angle_text_x, angle_text_y, f"{longest_edge_angle:.1f}°", ha='center', fontsize=12)
    
    # Add rotation information
    plt.figtext(0.5, 0.02, 
               f"Edge Angle: {longest_edge_angle:.2f}° | Target Angle: {target_angle}° | " 
               f"Rotation Needed: {rotation_angle:.2f}°",
               ha="center", fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Add final orientation indicator with new classification
    if target_angle == 0:
        orientation = "Vertical (North)"
    elif target_angle == 90:
        orientation = "Horizontal (East)"
    else:  # target_angle == 180
        orientation = "Vertical (South)"
    
    plt.figtext(0.5, 0.05, f"After rotation, this edge will be aligned: {orientation}", 
               ha="center", fontsize=12, weight='bold')
    
    # Add enhanced compass
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    compass_x = xmax - (xmax - xmin) * 0.1
    compass_y = ymax - (ymax - ymin) * 0.1
    compass_size = min(xmax - xmin, ymax - ymin) * 0.05
    
    # Draw compass with North, East, South
    plt.arrow(compass_x, compass_y, 0, compass_size, head_width=compass_size/3, 
             head_length=compass_size/3, fc='blue', ec='blue')
    plt.arrow(compass_x, compass_y, compass_size, 0, head_width=compass_size/3, 
             head_length=compass_size/3, fc='red', ec='red')
    plt.arrow(compass_x, compass_y, 0, -compass_size, head_width=compass_size/3, 
             head_length=compass_size/3, fc='purple', ec='purple')
    
    # Add labels
    plt.text(compass_x, compass_y + compass_size * 1.2, 'N', ha='center', va='center', fontsize=10)
    plt.text(compass_x + compass_size * 1.2, compass_y, 'E', ha='center', va='center', fontsize=10)
    plt.text(compass_x, compass_y - compass_size * 1.2, 'S', ha='center', va='center', fontsize=10)
    
    # Add title and labels
    plt.title(f"Building with Longest Edge (Angle with North: {longest_edge_angle:.2f}°)", fontsize=14)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the text at the bottom
    plt.show()

# Load geometry data
try:
    with open('data/geometry.json', 'r') as f:
        geo_data = json.load(f)
except FileNotFoundError:
    # Try without the data/ prefix if file not found
    with open('geometry.json', 'r') as f:
        geo_data = json.load(f)

# Process buildings
building_polygons = []
for building in geo_data.get('buildings', []):
    polygon = Polygon(building['coordinates'])
    if polygon.is_valid:
        building_polygons.append(polygon)
    else:
        # Try to fix invalid polygon
        polygon = polygon.buffer(0)
        building_polygons.append(polygon)

# Create a MultiPolygon for all buildings
building = MultiPolygon(building_polygons)

# Find the longest edge and its angle
longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly = find_longest_edge_and_angle(building)
print(f"Longest edge length: {longest_edge_length:.2f}, angle with north: {longest_edge_angle:.2f} degrees")

# Determine target angle with new classification
target_angle = get_target_angle(longest_edge_angle)
print(f"Target angle: {target_angle} degrees")

# Calculate rotation angle
rotation_angle = calculate_rotation_angle(longest_edge_angle, target_angle)
print(f"Rotation angle needed: {rotation_angle:.2f} degrees")

# Visualize
visualize_building_and_longest_edge(building, longest_edge_start, longest_edge_end, longest_edge_angle, longest_poly)