"""
Visualizer for plotting the building, graph, and optimized path.
"""
import os
import logging
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point
import numpy as np
from datetime import datetime

class PathVisualizer:
    """Visualizes the building, graph, and optimized path."""
    
    def __init__(self, config):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot(self, G, building, obstacles, segments, selected_edges, path_metrics=None):
        """
        Plot the building, graph, and optimized path.
        
        Args:
            G: networkx DiGraph
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            segments: List of segments
            selected_edges: List of selected edges
            path_metrics: Dictionary containing metrics about the path (optional)
        """
        self.logger.info("Creating visualization")
        
        # Create figure with larger size
        plt.figure(figsize=(18, 12))
        
        # Plot graph nodes and edges if enabled
        if self.config['visualization']['show_grid']:
            self._plot_graph(G, selected_edges)
        
        # Plot building and obstacles
        self._plot_geometries(building, obstacles)
        
        # Plot segments
        self._plot_segments(segments)
        
        # Plot selected path edges
        self._plot_selected_path(G, selected_edges)
        
        # Set dynamic plot title
        edge_size = self.config['graph']['grid_spacing']
        segment_size = self.config['visibility']['segment_size']
        particle_spacing = self.config['visibility']['particle_visibility']['spacing']
        
        title = f"Optimal Path with {edge_size}m Edges and {segment_size}m Segments"
        if self.config['visibility']['particle_visibility']['enabled']:
            title += f" and {particle_spacing}m Particle for Visibility Analysing"
        
        plt.title(title, fontsize=14)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add path metrics if provided (removing any rotation angle info)
        if path_metrics:
            metrics_text = f"Path Length: {path_metrics.get('path_length', 'N/A'):.2f} m\n"
            metrics_text += f"Selected Edges: {path_metrics.get('num_edges', 'N/A')}\n"
            metrics_text += f"VRF: {path_metrics.get('vrf', 'N/A'):.4f}"
            
            plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=12, 
                        bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save plot with timestamp
        if self.config['output']['plots']['save']:
            self._save_plot_with_timestamp()
        
        # Display plot if enabled
        if self.config['output']['plots']['display']:
            plt.show()
        else:
            plt.close()
    
    def _plot_graph(self, G, selected_edges):
        """
        Plot the graph nodes and edges.
        
        Args:
            G: networkx DiGraph
            selected_edges: List of selected edges
        """
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=self.config['visualization']['node_size'],
            node_color='lightblue',
            edgecolors='blue',
            alpha=0.7
        )
        
        # Plot edges (excluding selected edges which will be highlighted)
        non_selected_edges = [e for e in G.edges() if e not in selected_edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=non_selected_edges,
            width=self.config['visualization']['edge_width'],
            edge_color='gray',
            alpha=0.5,
            arrows=False
        )
        
        # Plot node labels if enabled
        if self.config['visualization'].get('show_node_ids', False):
            # Draw labels with configured font size
            nx.draw_networkx_labels(
                G, pos,
                font_size=self.config['visualization'].get('node_id_font_size', 6),
                font_family='sans-serif'
            )
    
    def _plot_geometries(self, building, obstacles):
        """
        Plot the building and obstacles.
        
        Args:
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
        """
        # Plot building
        for poly in building.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2)
            plt.fill(x, y, alpha=0.2, color='green')
            
            # Add building ID labels if enabled
            if self.config['visualization']['show_building_ids']:
                centroid = poly.centroid
                plt.text(
                    centroid.x, centroid.y,
                    f"Building",
                    fontsize=10,
                    ha='center',
                    color='darkgreen'
                )
        
        # Plot radiation obstacles
        if 'radiation' in obstacles and not obstacles['radiation'].is_empty:
            for poly in obstacles['radiation'].geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'r-', linewidth=2)
                plt.fill(x, y, alpha=0.6, color='red', hatch='///')
        
        # Plot visibility obstacles
        if 'visibility' in obstacles and not obstacles['visibility'].is_empty:
            for poly in obstacles['visibility'].geoms:
                if not any(poly.equals(rad_poly) for rad_poly in obstacles.get('radiation', {}).geoms):
                    x, y = poly.exterior.xy
                    plt.plot(x, y, 'k-', linewidth=2)
                    plt.fill(x, y, alpha=0.6, color='black')
    
    def _plot_segments(self, segments):
        """
        Plot the building segments.
        
        Args:
            segments: List of segments
        """
        # Plot segment endpoints
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            # Plot segment endpoints
            plt.plot(
                seg_start.x, seg_start.y,
                'ro',
                markersize=3
            )
            plt.plot(
                seg_end.x, seg_end.y,
                'ro',
                markersize=3
            )
            
            # Add segment ID labels if enabled
            if self.config['visualization']['show_segment_ids']:
                midpoint_x = (seg_start.x + seg_end.x) / 2.0
                midpoint_y = (seg_start.y + seg_end.y) / 2.0
                plt.text(
                    midpoint_x, midpoint_y,
                    str(seg_idx),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='blue',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                )
    
    def _plot_selected_path(self, G, selected_edges):
        """
        Plot the selected path.
        
        Args:
            G: networkx DiGraph
            selected_edges: List of selected edges
        """
        if not selected_edges:
            self.logger.warning("No selected edges to plot")
            return
        
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot selected edges as the final path
        nx.draw_networkx_edges(
            G, pos,
            edgelist=selected_edges,
            width=self.config['visualization']['selected_edge_width'],
            edge_color='red',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )
        
        # Add directions along the path
        if len(selected_edges) <= 30:  # Only for relatively small paths
            for i, j in selected_edges:
                # Calculate edge midpoint for direction indicator
                xi, yi = pos[i]
                xj, yj = pos[j]
                midpoint_x = (xi + xj) / 2.0
                midpoint_y = (yi + yj) / 2.0
                
                # Calculate direction
                dx = xj - xi
                dy = yj - yi
                
                # Normalize direction vector
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                # Plot direction arrow
                plt.arrow(
                    midpoint_x - dx*0.5, midpoint_y - dy*0.5,
                    dx*0.5, dy*0.5,
                    head_width=0.8,
                    head_length=1.0,
                    fc='blue',
                    ec='blue',
                    zorder=5
                )
    
    def _save_plot_with_timestamp(self):
        """Save the plot to a file with a timestamp."""
        # Create output directory if it doesn't exist
        output_path = self.config['output']['plots']['path']
        os.makedirs(output_path, exist_ok=True)
        
        # Get file format and dpi
        filename = self.config['output']['plots']['filename']
        base_name, extension = os.path.splitext(filename)
        if not extension:
            extension = f".{self.config['output']['plots']['format']}"
        
        # Create timestamped filename
        timestamped_filename = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
        
        # Create full path
        full_path = os.path.join(output_path, timestamped_filename)
        
        # Save the plot
        plt.savefig(full_path, format=self.config['output']['plots']['format'], 
                   dpi=self.config['output']['plots']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved plot to {full_path}")