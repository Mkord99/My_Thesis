"""
Graph builder for creating a grid-based graph for path planning.
"""
import logging
import numpy as np
import networkx as nx
from scipy.spatial import distance
from shapely.geometry import Point, LineString

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
    
    def build_graph(self, building, obstacles):
        """
        Build a grid-based graph based on the building and obstacles.
        
        Args:
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            Tuple of (networkx DiGraph, list of grid points)
        """
        self.logger.info("Building grid graph")
        
        # Create buffers around the building
        inner_buffer, outer_buffer = self._create_buffers(building)
        
        # Generate grid points
        grid_points = self._generate_grid_points(building, inner_buffer, outer_buffer, obstacles)
        self.logger.info(f"Generated {len(grid_points)} grid points")
        
        # Create graph
        G = self._create_graph(grid_points, inner_buffer, obstacles)
        self.logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # NOTE: Graph preprocessing is removed to preserve all nodes for visibility
        
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
    
    def _generate_grid_points(self, building, inner_buffer, outer_buffer, obstacles):
        """
        Generate grid points that will form the graph nodes.
        
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
        
        # Generate points in a grid
        grid_points = []
        for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
            for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
                point = Point(x, y)
                
                # Check if the point is in a valid location
                if (not building.contains(point) and  # Not inside building
                    outer_buffer.contains(point) and  # Inside outer buffer
                    not obstacles['radiation'].contains(point)):  # Not in radiation obstacle
                    
                    # Make sure it's not too close to the building
                    min_distance = min(poly.exterior.distance(point) for poly in inner_buffer.geoms)
                    if min_distance >= 0:
                        grid_points.append(point)
        
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
    
    # The preprocessing method is still defined but not used
    def _preprocess_graph(self, G):
        """
        Preprocess the graph to reduce complexity while preserving optimality.
        THIS METHOD IS KEPT FOR REFERENCE BUT NOT CURRENTLY USED.
        
        Args:
            G: networkx DiGraph
            
        Returns:
            Processed networkx DiGraph with reduced complexity
        """
        self.logger.info("Preprocessing graph to reduce complexity")
        
        # Step 1: Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            G.remove_nodes_from(isolated_nodes)
            self.logger.info(f"Removed {len(isolated_nodes)} isolated nodes")
        
        # Step 2: Simplify paths where a node has exactly one predecessor and one successor
        original_node_count = G.number_of_nodes()
        simplified_count = 0
        
        # Make a copy of the node list to avoid modification during iteration
        for node in list(G.nodes()):
            # Skip if node was already removed in a previous iteration
            if not G.has_node(node):
                continue
                
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            
            # Check if node has exactly one predecessor and one successor
            if len(predecessors) == 1 and len(successors) == 1:
                pred = predecessors[0]
                succ = successors[0]
                
                # Only simplify if predecessor and successor are different
                # and not already connected
                if pred != succ and not G.has_edge(pred, succ):
                    # Calculate new edge weight (preserve path length)
                    new_weight = G[pred][node]['weight'] + G[node][succ]['weight']
                    
                    # Add direct edge between predecessor and successor
                    G.add_edge(pred, succ, weight=new_weight)
                    
                    # Remove the intermediate node
                    G.remove_node(node)
                    simplified_count += 1
        
        if simplified_count > 0:
            self.logger.info(f"Simplified {simplified_count} path nodes")
            self.logger.info(f"Reduced graph from {original_node_count} to {G.number_of_nodes()} nodes")
        
        return G