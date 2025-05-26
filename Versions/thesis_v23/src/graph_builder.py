"""
Graph builder for creating a grid-based graph for path planning.
The grid can be aligned with the building's longest edge.
"""
import logging
import numpy as np
import networkx as nx
from scipy.spatial import distance
from shapely.geometry import Point, LineString
import math

class GraphBuilder:
    """Builds a grid-based graph for path planning."""
    
    def __init__(self, config):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rotation_enabled = config.get('rotation', {}).get('enabled', False)
        self.rotation_angle = 0
        self.rotation_center = (0, 0)
    
    def build_graph(self, building, obstacles, rotation_angle=None, rotation_center=None):
        """
        Build a grid-based graph based on the building and obstacles.
        If rotation is enabled, the grid will be aligned with the building's longest edge.
        
        Args:
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            rotation_angle: Angle to rotate the grid (opposite of building rotation)
            rotation_center: Center point for rotation
            
        Returns:
            Tuple of (networkx DiGraph, list of grid points)
        """
        self.logger.info("Building grid graph")
        
        # Store rotation parameters if provided
        if self.rotation_enabled and rotation_angle is not None:
            self.rotation_angle = -rotation_angle  # Use opposite rotation for grid
            self.rotation_center = rotation_center
            self.logger.info(f"Grid will be aligned with building orientation (rotated by {self.rotation_angle:.2f}°)")
        
        # Create buffers around the building
        inner_buffer, outer_buffer = self._create_buffers(building)
        
        # Generate grid points (possibly aligned with building)
        grid_points = self._generate_grid_points(building, inner_buffer, outer_buffer, obstacles)
        self.logger.info(f"Generated {len(grid_points)} grid points")
        
        # Create graph
        G = self._create_graph(grid_points, inner_buffer, obstacles)
        self.logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, grid_points
    
    def _create_buffers(self, building):
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
    
    def _rotate_point(self, point, inverse=False):
        """
        Rotate a point according to the grid rotation parameters.
        
        Args:
            point: Point to rotate (x, y)
            inverse: If True, apply reverse rotation
            
        Returns:
            Rotated point (x, y)
        """
        if not self.rotation_enabled or self.rotation_angle == 0:
            return point
        
        # Use the opposite angle if inverse rotation is requested
        angle = -self.rotation_angle if inverse else self.rotation_angle
        
        # Extract coordinates
        if isinstance(point, Point):
            x, y = point.x, point.y
        else:
            x, y = point
        
        # Get rotation center
        cx, cy = self.rotation_center
        
        # Translate to origin
        tx, ty = x - cx, y - cy
        
        # Rotate
        angle_rad = math.radians(angle)
        rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
        ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
        
        # Translate back
        result_x = rx + cx
        result_y = ry + cy
        
        return (result_x, result_y)
    
    def _generate_grid_points(self, building, inner_buffer, outer_buffer, obstacles):
        """
        Generate grid points that will form the graph nodes.
        If rotation is enabled, the grid will be aligned with the building's orientation.
        
        Args:
            building: MultiPolygon representing the building
            inner_buffer: Buffer around the building (closer)
            outer_buffer: Buffer around the building (outer)
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            List of valid grid points (shapely Points)
        """
        grid_spacing = self.config['graph']['grid_spacing']
        
        # Get the bounds of the outer buffer
        xmin, ymin, xmax, ymax = outer_buffer.bounds
        
        # If rotation is enabled, we need to make the grid larger to ensure coverage
        if self.rotation_enabled and self.rotation_angle != 0:
            # Calculate diagonal length
            diagonal = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            # Find center
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            # Expand bounds to ensure coverage after rotation
            xmin = center_x - diagonal / 2
            xmax = center_x + diagonal / 2
            ymin = center_y - diagonal / 2
            ymax = center_y + diagonal / 2
        
        # Generate points in a grid
        grid_points = []
        
        for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
            for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
                # If rotation is enabled, rotate the candidate point
                if self.rotation_enabled and self.rotation_angle != 0:
                    # Start with a rotated grid point
                    grid_x, grid_y = self._rotate_point((x, y), inverse=True)
                    candidate_point = Point(grid_x, grid_y)
                else:
                    candidate_point = Point(x, y)
                
                # Check if the point is in a valid location
                if (not building.contains(candidate_point) and  # Not inside building
                    outer_buffer.contains(candidate_point) and  # Inside outer buffer
                    not obstacles['radiation'].contains(candidate_point)):  # Not in radiation obstacle
                    
                    # Make sure it's not too close to the building
                    min_distance = min(poly.exterior.distance(candidate_point) for poly in inner_buffer.geoms)
                    if min_distance >= 0:
                        grid_points.append(candidate_point)
        
        return grid_points
    
    def _create_graph(self, grid_points, inner_buffer, obstacles):
        """
        Create a graph from grid points.
        
        Args:
            grid_points: List of grid points
            inner_buffer: Buffer around the building (closer)
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            networkx DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i, point in enumerate(grid_points):
            G.add_node(i, pos=(point.x, point.y))
        
        # Add edges
        max_edge_distance = self.config['graph']['max_edge_distance'] * np.sqrt(2)
        
        for i, p1 in enumerate(grid_points):
            for j, p2 in enumerate(grid_points):
                if i != j:
                    # Calculate distance between points
                    dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
                    
                    # Check if points are within the maximum edge distance
                    if dist <= max_edge_distance:
                        # Create a line between the points
                        edge_line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                        
                        # Ensure edge does not intersect buffer or obstacles
                        if (not inner_buffer.intersects(edge_line) and 
                            not obstacles['radiation'].intersects(edge_line)):
                            G.add_edge(i, j, weight=dist)
        
        # Verify the graph is connected, otherwise log a warning
        if not nx.is_weakly_connected(G):
            self.logger.warning("The generated graph is not connected.")
        
        return G