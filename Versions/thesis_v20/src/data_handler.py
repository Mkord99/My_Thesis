"""
Data handler for loading and processing geometry data.
"""
import json
import logging
import numpy as np
import math
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.affinity import rotate, translate
from scipy.spatial import transform
import matplotlib.pyplot as plt

class GeometryLoader:
    """Loads and processes geometry data from configuration files."""
    
    def __init__(self, config):
        """
        Initialize the geometry loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rotation_angle = 0  # Store the rotation angle for later use
        self.rotation_origin = None  # Store the rotation origin for later use
        self.rotated = False  # Flag to indicate if geometries are rotated
    
    def load_geometries(self):
        """
        Load building and obstacle geometries from the specified file.
        
        Returns:
            Tuple of (building MultiPolygon, obstacles MultiPolygon, dictionary of all polygons)
        """
        geometry_file = self.config['data']['geometry_file']
        self.logger.info(f"Loading geometries from {geometry_file}")
        
        try:
            with open(geometry_file, 'r') as f:
                geo_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load geometry file: {e}")
            raise
        
        # Process buildings
        building_polygons = []
        for building in geo_data.get('buildings', []):
            try:
                polygon = Polygon(building['coordinates'])
                if not polygon.is_valid:
                    self.logger.warning(f"Building {building.get('id', 'unknown')} has an invalid polygon")
                    polygon = polygon.buffer(0)  # Try to fix invalid polygon
                building_polygons.append(polygon)
            except Exception as e:
                self.logger.error(f"Error processing building {building.get('id', 'unknown')}: {e}")
                raise
        
        # Create a MultiPolygon for all buildings
        building = MultiPolygon(building_polygons)
        
        # Process obstacles
        radiation_obstacles = []
        visibility_obstacles = []
        all_polygons = {'buildings': building_polygons}
        
        for obstacle in geo_data.get('obstacles', []):
            try:
                polygon = Polygon(obstacle['coordinates'])
                if not polygon.is_valid:
                    self.logger.warning(f"Obstacle {obstacle.get('id', 'unknown')} has an invalid polygon")
                    polygon = polygon.buffer(0)  # Try to fix invalid polygon
                
                # Check obstacle type
                obstacle_type = obstacle.get('type', [])
                if not isinstance(obstacle_type, list):
                    obstacle_type = [obstacle_type]
                
                if 'radiation' in obstacle_type:
                    radiation_obstacles.append(polygon)
                
                if 'visibility' in obstacle_type:
                    visibility_obstacles.append(polygon)
                    
            except Exception as e:
                self.logger.error(f"Error processing obstacle {obstacle.get('id', 'unknown')}: {e}")
                raise
        
        # Create MultiPolygons for obstacles
        radiation_obstacles_multi = MultiPolygon(radiation_obstacles) if radiation_obstacles else MultiPolygon([])
        visibility_obstacles_multi = MultiPolygon(visibility_obstacles) if visibility_obstacles else MultiPolygon([])
        
        all_polygons = {
            'buildings': building_polygons,
            'radiation_obstacles': radiation_obstacles,
            'visibility_obstacles': visibility_obstacles
        }
        
        # Store original geometries
        self.original_building = building
        self.original_obstacles = {
            'radiation': radiation_obstacles_multi,
            'visibility': visibility_obstacles_multi
        }
        self.original_polygons = all_polygons
        
        # Return the building, obstacles and all polygons
        return building, {
            'radiation': radiation_obstacles_multi,
            'visibility': visibility_obstacles_multi
        }, all_polygons
    
    def create_buffers(self, building):
        """
        Create buffer zones around the building.
        
        Args:
            building: MultiPolygon representing the building
            
        Returns:
            Tuple of (inner buffer, outer buffer)
        """
        inner_distance = self.config['data']['buffer_distances']['inner']
        outer_distance = self.config['data']['buffer_distances']['outer']
        
        inner_buffer = building.buffer(inner_distance)
        outer_buffer = building.buffer(outer_distance)
        
        return inner_buffer, outer_buffer
    
    def find_longest_edge_and_angle(self, building):
        """
        Find the longest edge in the building MultiPolygon and its angle with north.
        
        Args:
            building: MultiPolygon representing the building
            
        Returns:
            Tuple of (longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly)
        """
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
        
        # For debug purposes, visualize the longest edge
        self._debug_visualize_longest_edge(building, longest_edge_start, longest_edge_end, longest_edge_angle)
        
        return longest_edge_length, longest_edge_angle, longest_edge_start, longest_edge_end, longest_poly
    
    def _debug_visualize_longest_edge(self, building, longest_edge_start, longest_edge_end, angle):
        """Debug visualization of the longest edge and its angle"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Plot building
            for poly in building.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'g-', linewidth=2)
                plt.fill(x, y, alpha=0.2, color='green')
            
            # Plot longest edge
            if longest_edge_start and longest_edge_end:
                plt.plot([longest_edge_start.x, longest_edge_end.x], 
                         [longest_edge_start.y, longest_edge_end.y], 
                         'r-', linewidth=3, label=f"Longest Edge: Angle {angle:.2f}°")
            
                # Add labels
                plt.title(f"Longest Building Edge (Angle with North: {angle:.2f}°)")
                plt.xlabel("X-coordinate")
                plt.ylabel("Y-coordinate")
                plt.grid(True)
                plt.legend()
                
                # Save to debug file
                plt.savefig('output/debug_longest_edge.png')
                plt.close()
                self.logger.info("Saved debug visualization of longest edge to output/debug_longest_edge.png")
        except Exception as e:
            self.logger.error(f"Error creating debug visualization: {e}")
            # Don't let visualization errors stop the process
            pass
    
    def get_target_angle(self, longest_edge_angle):
        """
        Determine target angle based on the specified classification:
        If angle is between 0° and 45°, target is 0° (North)
        If angle is between 45° and 135°, target is 90° (East)
        If angle is between 135° and 180°, target is 180° (South)
        
        Args:
            longest_edge_angle: Angle of the longest edge with north in degrees
            
        Returns:
            Target angle in degrees
        """
        if 0 <= longest_edge_angle < 45:
            return 0  # Align with North
        elif 45 <= longest_edge_angle < 135:
            return 90  # Align with East
        else:  # 135 <= longest_edge_angle <= 180
            return 180  # Align with South
    
    def calculate_rotation_angle(self, longest_edge_angle, target_angle):
        """
        Calculate the rotation angle needed to align the longest edge with the target angle.
        
        Args:
            longest_edge_angle: Angle of the longest edge with north in degrees
            target_angle: Target angle in degrees
            
        Returns:
            Rotation angle in degrees
        """
        # Calculate the difference between current and target angles
        rotation_angle = target_angle - longest_edge_angle
        
        # Normalize to range [-180, 180] for most efficient rotation
        while rotation_angle > 180:
            rotation_angle -= 360
        while rotation_angle < -180:
            rotation_angle += 360
            
        return rotation_angle
    
    def rotate_coordinates(self, x, y, angle_degrees, origin_x, origin_y):
        """
        Rotate coordinates around an origin point by the given angle.
        
        Args:
            x, y: Coordinates to rotate
            angle_degrees: Rotation angle in degrees
            origin_x, origin_y: Origin point for rotation
            
        Returns:
            Tuple of rotated (x, y) coordinates
        """
        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)
        
        # Translate point to origin
        x_shifted = x - origin_x
        y_shifted = y - origin_y
        
        # Rotate point
        x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
        y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
        
        # Translate back
        x_final = x_rotated + origin_x
        y_final = y_rotated + origin_y
        
        return x_final, y_final
    
    def _rotate_polygon(self, polygon, angle, origin):
        """
        Manually rotate a polygon using the rotation matrix approach.
        
        Args:
            polygon: Shapely Polygon
            angle: Rotation angle in degrees
            origin: Tuple (x, y) for rotation origin
            
        Returns:
            Rotated polygon
        """
        coords = list(polygon.exterior.coords)
        rotated_coords = []
        
        for x, y in coords:
            x_rot, y_rot = self.rotate_coordinates(x, y, angle, origin[0], origin[1])
            rotated_coords.append((x_rot, y_rot))
        
        # Handle interior rings (holes)
        rotated_interiors = []
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            rotated_interior = []
            for x, y in interior_coords:
                x_rot, y_rot = self.rotate_coordinates(x, y, angle, origin[0], origin[1])
                rotated_interior.append((x_rot, y_rot))
            rotated_interiors.append(rotated_interior)
        
        # Create new polygon
        rotated_poly = Polygon(rotated_coords, rotated_interiors)
        return rotated_poly
    
    def rotate_geometry(self, geometry, angle, origin):
        """
        Rotate a geometry using custom rotation implementation.
        
        Args:
            geometry: Shapely geometry to rotate
            angle: Rotation angle in degrees
            origin: Tuple (x,y) representing the rotation origin
            
        Returns:
            Rotated geometry
        """
        if geometry is None or geometry.is_empty:
            return geometry
        
        if isinstance(geometry, Point):
            x_rot, y_rot = self.rotate_coordinates(
                geometry.x, geometry.y, angle, origin[0], origin[1]
            )
            return Point(x_rot, y_rot)
        
        elif isinstance(geometry, Polygon):
            return self._rotate_polygon(geometry, angle, origin)
        
        elif isinstance(geometry, MultiPolygon):
            rotated_polys = []
            for poly in geometry.geoms:
                rotated_poly = self._rotate_polygon(poly, angle, origin)
                rotated_polys.append(rotated_poly)
            return MultiPolygon(rotated_polys)
        
        elif isinstance(geometry, LineString):
            coords = list(geometry.coords)
            rotated_coords = []
            for x, y in coords:
                x_rot, y_rot = self.rotate_coordinates(x, y, angle, origin[0], origin[1])
                rotated_coords.append((x_rot, y_rot))
            return LineString(rotated_coords)
        
        else:
            # Fallback to shapely's rotate for other geometry types
            return rotate(geometry, angle, origin=origin, use_radians=False)
    
    def rotate_all_geometries(self, building, obstacles, rotation_angle):
        """
        Rotate all geometries (building and obstacles) by the specified angle.
        
        Args:
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            rotation_angle: Rotation angle in degrees
            
        Returns:
            Tuple of (rotated_building, rotated_obstacles)
        """
        # Calculate a global rotation origin (centroid of all geometries)
        all_geometries = [building]
        for obs_type, obs_geom in obstacles.items():
            if not obs_geom.is_empty:
                all_geometries.append(obs_geom)
        
        # Calculate a bounding box center as rotation origin
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for geom in all_geometries:
            geom_bounds = geom.bounds
            min_x = min(min_x, geom_bounds[0])
            min_y = min(min_y, geom_bounds[1])
            max_x = max(max_x, geom_bounds[2])
            max_y = max(max_y, geom_bounds[3])
        
        # Use center of bounding box as rotation origin
        origin_x = (min_x + max_x) / 2
        origin_y = (min_y + max_y) / 2
        self.rotation_origin = (origin_x, origin_y)
        
        # Store rotation angle for later use
        self.rotation_angle = rotation_angle
        
        # Debug output
        self.logger.info(f"Rotating all geometries by {rotation_angle:.2f} degrees around origin ({origin_x:.2f}, {origin_y:.2f})")
        
        # Rotate building
        rotated_building = self.rotate_geometry(building, rotation_angle, self.rotation_origin)
        
        # Rotate obstacles
        rotated_obstacles = {}
        for obs_type, obs_geom in obstacles.items():
            if not obs_geom.is_empty:
                rotated_obstacles[obs_type] = self.rotate_geometry(obs_geom, rotation_angle, self.rotation_origin)
            else:
                rotated_obstacles[obs_type] = obs_geom
        
        # Set rotation flag
        self.rotated = True
        
        # Debug - save a visualization of before and after rotation
        self._debug_visualize_rotation(building, rotated_building, obstacles, rotated_obstacles)
        
        return rotated_building, rotated_obstacles
    
    def _debug_visualize_rotation(self, orig_building, rot_building, orig_obstacles, rot_obstacles):
        """Create a debug visualization of the rotation effect"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot original geometries
            for poly in orig_building.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'g-', linewidth=1, alpha=0.5)
                plt.fill(x, y, alpha=0.1, color='green')
            
            for obs_type, obs_multi in orig_obstacles.items():
                if not obs_multi.is_empty:
                    for poly in obs_multi.geoms:
                        x, y = poly.exterior.xy
                        color = 'red' if obs_type == 'radiation' else 'blue'
                        plt.plot(x, y, f'{color}-', linewidth=1, alpha=0.5)
                        plt.fill(x, y, alpha=0.1, color=color)
            
            # Plot rotated geometries with stronger colors
            for poly in rot_building.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'g-', linewidth=2)
                plt.fill(x, y, alpha=0.3, color='green')
            
            for obs_type, obs_multi in rot_obstacles.items():
                if not obs_multi.is_empty:
                    for poly in obs_multi.geoms:
                        x, y = poly.exterior.xy
                        color = 'red' if obs_type == 'radiation' else 'blue'
                        plt.plot(x, y, f'{color}-', linewidth=2)
                        plt.fill(x, y, alpha=0.3, color=color)
            
            # Mark rotation origin
            plt.plot(self.rotation_origin[0], self.rotation_origin[1], 'ko', markersize=10)
            plt.text(self.rotation_origin[0], self.rotation_origin[1], "Rotation Origin", 
                    fontsize=12, ha='center', va='bottom')
            
            # Add title and labels
            plt.title(f"Geometry Rotation by {self.rotation_angle:.2f} degrees")
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.grid(True)
            plt.axis('equal')
            
            # Save to debug file
            plt.savefig('output/debug_rotation.png')
            plt.close()
            self.logger.info("Saved debug visualization of rotation to output/debug_rotation.png")
        except Exception as e:
            self.logger.error(f"Error creating debug rotation visualization: {e}")
            # Don't let visualization errors stop the process
    
    def rotate_point(self, point, reverse=False):
        """
        Rotate a point (in place or reverse the rotation).
        
        Args:
            point: Point object or tuple (x,y)
            reverse: If True, rotate in the opposite direction
            
        Returns:
            Rotated Point object
        """
        if not hasattr(self, 'rotation_angle') or not hasattr(self, 'rotation_origin'):
            self.logger.warning("Cannot rotate point: rotation parameters not set")
            return point
        
        angle = -self.rotation_angle if reverse else self.rotation_angle
        
        if isinstance(point, tuple):
            x, y = point
            x_rot, y_rot = self.rotate_coordinates(x, y, angle, self.rotation_origin[0], self.rotation_origin[1])
            return (x_rot, y_rot)
        else:
            x_rot, y_rot = self.rotate_coordinates(point.x, point.y, angle, self.rotation_origin[0], self.rotation_origin[1])
            return Point(x_rot, y_rot)
    
    def rotate_coordinates_list(self, coords_list, reverse=False):
        """
        Rotate a list of coordinate tuples.
        
        Args:
            coords_list: List of (x,y) coordinate tuples
            reverse: If True, rotate in the opposite direction
            
        Returns:
            List of rotated coordinate tuples
        """
        angle = -self.rotation_angle if reverse else self.rotation_angle
        rotated_coords = []
        
        for x, y in coords_list:
            x_rot, y_rot = self.rotate_coordinates(x, y, angle, self.rotation_origin[0], self.rotation_origin[1])
            rotated_coords.append((x_rot, y_rot))
        
        return rotated_coords
    
    def rotate_back_to_original(self, nodes_positions, selected_edges):
        """
        Rotate node positions and selected edges back to the original orientation.
        
        Args:
            nodes_positions: Dictionary mapping node IDs to (x,y) positions
            selected_edges: List of selected edges (node_id pairs)
            
        Returns:
            Tuple of (rotated_positions, rotated_edges)
        """
        if not self.rotated or self.rotation_angle == 0:
            return nodes_positions, selected_edges
        
        # Rotate node positions back
        rotated_positions = {}
        for node_id, pos in nodes_positions.items():
            rotated_pos = self.rotate_point(pos, reverse=True)
            rotated_positions[node_id] = rotated_pos
        
        # Edge references don't need to be rotated as they reference nodes
        # which are already rotated through their positions
        
        return rotated_positions, selected_edges