"""
Visualizer for plotting the building, graph, and optimized path.
"""
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
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
        self.rotation_enabled = config.get('rotation', {}).get('enabled', False)
    
    def plot(self, G, building, obstacles, segments, selected_edges, path_metrics=None, 
             rotation_params=None):
        """
        Plot the building, graph, and optimized path.
        
        Args:
            G: networkx DiGraph
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            segments: List of segments
            selected_edges: List of selected edges
            path_metrics: Dictionary containing metrics about the path (optional)
            rotation_params: Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle)
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
        self._plot_segments_on_building(building, segments)
        
        # Plot selected path edges
        self._plot_selected_path(G, selected_edges)
        
        # Set dynamic plot title
        edge_size = self.config['graph']['grid_spacing']
        segment_size = self.config['visibility']['segment_size']
        particle_spacing = self.config['visibility']['particle_visibility']['spacing']
        
        title = f"Optimal Path with {edge_size}m Edges and {segment_size}m Segments"
        if self.config['visibility']['particle_visibility']['enabled']:
            title += f" and {particle_spacing}m Particle for Visibility Analysing"
        
        # Add orientation information if available
        if self.rotation_enabled and rotation_params:
            rotation_angle, rotation_center, longest_edge_angle, target_angle = rotation_params
            title += f" (Grid Aligned with Building Orientation)"
        
        plt.title(title, fontsize=14)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add path metrics if provided
        if path_metrics:
            metrics_text = f"Path Length: {path_metrics.get('path_length', 'N/A'):.2f} m\n"
            metrics_text += f"Selected Edges: {path_metrics.get('num_edges', 'N/A')}\n"
            metrics_text += f"VRF: {path_metrics.get('vrf', 'N/A'):.4f}"
            
            # Add orientation information if enabled
            if self.rotation_enabled and rotation_params:
                rotation_angle, rotation_center, longest_edge_angle, target_angle = rotation_params
                metrics_text += f"\nLongest Edge Angle: {longest_edge_angle:.2f}°, Grid Aligned to {target_angle}°"
            
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
    
    def _plot_edge_vrf_heatmap(self, G, building, obstacles, vrf, output_dir):
        """
        Heatmap 3: Edge VRF (Visibility Ratio Factor) intensity on graph network.
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate VRF values for each edge
        edge_vrf_values = {}
        max_vrf = 0
        min_vrf = float('inf')
        
        for edge, vrf_value in vrf.items():
            edge_vrf_values[edge] = vrf_value
            max_vrf = max(max_vrf, vrf_value)
            min_vrf = min(min_vrf, vrf_value)
        
        self.logger.info(f"Edge VRF range: {min_vrf:.4f} to {max_vrf:.4f}")
        
        # Plot building and obstacles first
        self._plot_base_geometries(building, obstacles, show_legend=True)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create colormap for edges
        if max_vrf > min_vrf:
            cmap = plt.cm.viridis  # Blue to yellow colormap (good for VRF)
            norm = mcolors.Normalize(vmin=min_vrf, vmax=max_vrf)
            
            # Plot edges with colors based on VRF value
            for edge in G.edges():
                if edge in edge_vrf_values:
                    vrf_value = edge_vrf_values[edge]
                    color = cmap(norm(vrf_value))
                    
                    # Line width based on VRF (higher VRF = thicker line)
                    line_width = 1 + 3 * ((vrf_value - min_vrf) / (max_vrf - min_vrf))
                    alpha = 0.4 + 0.6 * ((vrf_value - min_vrf) / (max_vrf - min_vrf))
                    
                    # Get edge coordinates
                    start_pos = pos[edge[0]]
                    end_pos = pos[edge[1]]
                    
                    # Plot edge with color intensity
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color=color, linewidth=line_width, alpha=alpha, zorder=2)
            
            # Plot nodes with VRF-based coloring
            node_vrf_values = []
            for node in G.nodes():
                # Calculate node VRF as average of all incident edges
                node_vrf = 0
                edge_count = 0
                for edge in G.edges():
                    if node in edge and edge in edge_vrf_values:
                        node_vrf += edge_vrf_values[edge]
                        edge_count += 1
                
                if edge_count > 0:
                    node_vrf = node_vrf / edge_count
                node_vrf_values.append(node_vrf)
            
            # Normalize node colors
            if max(node_vrf_values) > min(node_vrf_values):
                nx.draw_networkx_nodes(G, pos, node_color=node_vrf_values, cmap=cmap, 
                                     node_size=50, alpha=0.8, 
                                     vmin=min(node_vrf_values), vmax=max(node_vrf_values))
            else:
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50, alpha=0.8)
            
            # Create colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
            cbar.set_label('VRF (Visibility Ratio Factor)', rotation=270, labelpad=20, fontsize=12)
            
            # Add statistics text box
            stats_text = f"VRF Statistics:\nMin: {min_vrf:.4f}\nMax: {max_vrf:.4f}\nRange: {max_vrf - min_vrf:.4f}"
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        else:
            # No VRF variation available
            nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50, alpha=0.8)
            self.logger.warning("No VRF variation available for heatmap")
        
        plt.title("Edge VRF (Visibility Ratio Factor) Heatmap\n(VRF = Visible Segments / Edge Length)", 
                 fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"3_edge_vrf_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['visibility_heatmaps']['format'], 
                   dpi=self.config['output']['visibility_heatmaps']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved edge VRF heatmap to {filename}")
        plt.close()
    
    def _plot_vrf_with_optimized_path(self, G, building, obstacles, vrf, selected_edges, output_dir):
        """
        Heatmap 4: VRF background with optimized path overlay to analyze path quality.
        """
        plt.figure(figsize=(16, 12))
        
        # Calculate VRF values for each edge
        edge_vrf_values = {}
        max_vrf = 0
        min_vrf = float('inf')
        
        for edge, vrf_value in vrf.items():
            edge_vrf_values[edge] = vrf_value
            max_vrf = max(max_vrf, vrf_value)
            min_vrf = min(min_vrf, vrf_value)
        
        self.logger.info(f"VRF with path overlay - VRF range: {min_vrf:.4f} to {max_vrf:.4f}")
        
        # Plot building and obstacles first
        self._plot_base_geometries(building, obstacles, show_legend=False)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create colormap for VRF background
        if max_vrf > min_vrf:
            cmap = plt.cm.viridis  # Blue to yellow colormap
            norm = mcolors.Normalize(vmin=min_vrf, vmax=max_vrf)
            
            # STEP 1: Plot ALL edges as VRF heatmap background
            for edge in G.edges():
                if edge in edge_vrf_values:
                    vrf_value = edge_vrf_values[edge]
                    color = cmap(norm(vrf_value))
                    
                    # Get edge coordinates
                    start_pos = pos[edge[0]]
                    end_pos = pos[edge[1]]
                    
                    # Plot background edge with VRF color
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color=color, linewidth=2, alpha=0.7, zorder=2)
            
            # STEP 2: Plot all nodes as small gray dots
            nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=20, 
                                 alpha=0.6, zorder=3)
            
            # STEP 3: Overlay the SELECTED PATH on top
            if selected_edges:
                # Plot selected path as thick red/black lines
                for edge in selected_edges:
                    start_pos = pos[edge[0]]
                    end_pos = pos[edge[1]]
                    
                    # Black outline for visibility
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color='black', linewidth=6, alpha=0.8, zorder=5)
                    # Red center line
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color='red', linewidth=4, alpha=1.0, zorder=6)
                
                # Highlight selected nodes
                selected_nodes = set()
                for edge in selected_edges:
                    selected_nodes.add(edge[0])
                    selected_nodes.add(edge[1])
                
                if selected_nodes:
                    selected_pos = {node: pos[node] for node in selected_nodes}
                    nx.draw_networkx_nodes(G.subgraph(selected_nodes), selected_pos, 
                                         node_color='red', node_size=60, alpha=0.9, 
                                         zorder=7, edgecolors='black', linewidths=2)
                
                # Add direction arrows for the path
                if len(selected_edges) <= 30:  # Only for smaller paths
                    for edge in selected_edges:
                        start_pos = pos[edge[0]]
                        end_pos = pos[edge[1]]
                        
                        # Arrow at 2/3 along the edge
                        arrow_x = start_pos[0] + 0.67 * (end_pos[0] - start_pos[0])
                        arrow_y = start_pos[1] + 0.67 * (end_pos[1] - start_pos[1])
                        
                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]
                        length = np.sqrt(dx**2 + dy**2)
                        
                        if length > 0:
                            dx, dy = dx/length, dy/length
                            plt.arrow(arrow_x - dx*1, arrow_y - dy*1, dx*2, dy*2,
                                    head_width=1.2, head_length=1.5, 
                                    fc='white', ec='black', linewidth=1,
                                    alpha=0.9, zorder=8)
            
            # Create colorbar for VRF values
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
            cbar.set_label('VRF (Visibility Ratio Factor)', rotation=270, labelpad=20, fontsize=12)
            
            # Calculate path VRF statistics
            if selected_edges:
                selected_vrf_values = [edge_vrf_values[edge] for edge in selected_edges if edge in edge_vrf_values]
                if selected_vrf_values:
                    avg_path_vrf = np.mean(selected_vrf_values)
                    min_path_vrf = min(selected_vrf_values)
                    max_path_vrf = max(selected_vrf_values)
                    avg_all_vrf = np.mean(list(edge_vrf_values.values()))
                    
                    # Performance comparison
                    performance_ratio = (avg_path_vrf / avg_all_vrf) if avg_all_vrf > 0 else 0
                    
                    stats_text = f"Path Performance Analysis:\n"
                    stats_text += f"Selected Edges: {len(selected_edges)}\n"
                    stats_text += f"Path Avg VRF: {avg_path_vrf:.4f}\n"
                    stats_text += f"Network Avg VRF: {avg_all_vrf:.4f}\n"
                    stats_text += f"Performance Ratio: {performance_ratio:.2f}x\n"
                    stats_text += f"Path VRF Range: {min_path_vrf:.4f} - {max_path_vrf:.4f}"
                    
                    # Color the stats box based on performance
                    box_color = "lightgreen" if performance_ratio > 1.0 else "lightyellow" if performance_ratio > 0.8 else "lightcoral"
                    
                    plt.figtext(0.02, 0.02, stats_text, fontsize=11, 
                               bbox={"facecolor": box_color, "alpha": 0.9, "pad": 10})
            
            # Create legend
            legend_elements = [
                plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, label='Low VRF Edges'),
                plt.Line2D([0], [0], color='yellow', linewidth=2, alpha=0.7, label='High VRF Edges'),
                plt.Line2D([0], [0], color='red', linewidth=4, label='Selected Path'),
                plt.scatter([0], [0], c='white', s=100, marker='>', edgecolors='black', label='Path Direction'),
                plt.scatter([0], [0], c='red', s=60, edgecolors='black', linewidths=2, label='Path Nodes')
            ]
            
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
        else:
            # Fallback if no VRF variation
            nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
            if selected_edges:
                nx.draw_networkx_edges(G, pos, edgelist=selected_edges, 
                                     edge_color='red', width=4, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=40, alpha=0.8)
            self.logger.warning("No VRF variation available for path overlay analysis")
        
        plt.title("Path Quality Analysis: Selected Route vs VRF Distribution\n" +
                 "(Red path shows optimization results over VRF heatmap background)", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"4_path_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['visibility_heatmaps']['format'], 
                   dpi=self.config['output']['visibility_heatmaps']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved path quality analysis heatmap to {filename}")
        plt.close()
    
    def create_visibility_heatmaps(self, G, building, obstacles, segments, edge_visibility, segment_visibility, vrf, selected_edges=None):
        """
        Create visibility heatmaps for analysis.
        
        Args:
            G: networkx DiGraph
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            segments: List of segments
            edge_visibility: Dictionary mapping edges to segments they can see
            segment_visibility: Dictionary mapping segments to edges that can see them
            vrf: Dictionary of Visibility Ratio Factor (VRF) for each edge
            selected_edges: List of selected edges from optimization (optional)
        """
        # Check if visibility heatmaps are enabled
        if not self.config['output'].get('visibility_heatmaps', {}).get('enabled', False):
            self.logger.info("Visibility heatmaps are disabled in config")
            return
        
        self.logger.info("Creating visibility heatmaps")
        
        # Create output directory
        heatmaps_dir = self.config['output']['visibility_heatmaps']['path']
        os.makedirs(heatmaps_dir, exist_ok=True)
        
        # Heatmap 1: Edge visibility intensity on graph network
        self._plot_edge_visibility_heatmap(G, building, obstacles, edge_visibility, heatmaps_dir)
        
        # Heatmap 2: Segment visibility intensity on building footprint
        self._plot_segment_visibility_heatmap(building, obstacles, segments, segment_visibility, heatmaps_dir)
        
        # Heatmap 3: Edge VRF intensity on graph network
        self._plot_edge_vrf_heatmap(G, building, obstacles, vrf, heatmaps_dir)
        
        # Heatmap 4: Edge VRF with optimized path overlay (only if optimization was run)
        if selected_edges is not None and len(selected_edges) > 0:
            self._plot_vrf_with_optimized_path(G, building, obstacles, vrf, selected_edges, heatmaps_dir)
    
    def _plot_edge_visibility_heatmap(self, G, building, obstacles, edge_visibility, output_dir):
        """
        Heatmap 1: Edge visibility intensity on graph network.
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate visibility counts for each edge
        edge_visibility_counts = {}
        max_visibility = 0
        
        for edge, visible_segments in edge_visibility.items():
            count = len(visible_segments)
            edge_visibility_counts[edge] = count
            max_visibility = max(max_visibility, count)
        
        self.logger.info(f"Edge visibility range: 0 to {max_visibility} segments")
        
        # Plot building and obstacles first
        self._plot_base_geometries(building, obstacles, show_legend=True)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create colormap for edges
        if max_visibility > 0:
            cmap = plt.cm.YlOrRd  # Yellow to red colormap
            norm = mcolors.Normalize(vmin=0, vmax=max_visibility)
            
            # Plot edges with colors based on visibility count
            for edge in G.edges():
                if edge in edge_visibility_counts:
                    visibility_count = edge_visibility_counts[edge]
                    color = cmap(norm(visibility_count))
                    alpha = 0.3 + 0.7 * (visibility_count / max_visibility)  # Variable transparency
                    
                    # Get edge coordinates
                    start_pos = pos[edge[0]]
                    end_pos = pos[edge[1]]
                    
                    # Plot edge with color intensity
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                            color=color, linewidth=2, alpha=alpha, zorder=2)
            
            # Plot nodes
            node_colors = []
            for node in G.nodes():
                # Calculate node visibility as sum of all incident edges
                node_visibility = 0
                for edge in G.edges():
                    if node in edge and edge in edge_visibility_counts:
                        node_visibility += edge_visibility_counts[edge]
                node_colors.append(node_visibility)
            
            # Normalize node colors
            if max(node_colors) > 0:
                node_norm = mcolors.Normalize(vmin=0, vmax=max(node_colors))
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, 
                                     node_size=40, alpha=0.8, vmin=0, vmax=max(node_colors))
            else:
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=40, alpha=0.8)
            
            # Create colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
            cbar.set_label('Number of Visible Segments', rotation=270, labelpad=20, fontsize=12)
        else:
            # No visibility data available
            nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=40, alpha=0.8)
            self.logger.warning("No edge visibility data available for heatmap")
        
        plt.title("Edge Visibility Heatmap\n(Number of Segments Visible from Each Edge)", 
                 fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"1_edge_visibility_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['visibility_heatmaps']['format'], 
                   dpi=self.config['output']['visibility_heatmaps']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved edge visibility heatmap to {filename}")
        plt.close()
    
    def _plot_segment_visibility_heatmap(self, building, obstacles, segments, segment_visibility, output_dir):
        """
        Heatmap 2: Segment visibility intensity on building footprint.
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate visibility counts for each segment
        segment_visibility_counts = {}
        max_segment_visibility = 0
        
        for seg_idx, visible_edges in segment_visibility.items():
            count = len(visible_edges)
            segment_visibility_counts[seg_idx] = count
            max_segment_visibility = max(max_segment_visibility, count)
        
        self.logger.info(f"Segment visibility range: 0 to {max_segment_visibility} edges")
        
        # Plot obstacles first (without building)
        self._plot_obstacles_only(obstacles)
        
        # Create colormap for segments
        if max_segment_visibility > 0:
            cmap = plt.cm.plasma  # Purple to yellow colormap
            norm = mcolors.Normalize(vmin=0, vmax=max_segment_visibility)
            
            # Plot building outline
            for poly in building.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'black', linewidth=2, alpha=0.8)
            
            # Plot segments following building boundary with colors based on visibility count
            segment_size = self.config['visibility']['segment_size']
            boundary_lines = [poly.exterior for poly in building.geoms]
            
            segment_idx = 0
            for boundary_line in boundary_lines:
                current_distance = 0
                boundary_length = boundary_line.length
                
                while current_distance < boundary_length:
                    # Calculate segment end distance
                    seg_end_distance = min(current_distance + segment_size, boundary_length)
                    
                    # Get start and end points on the boundary
                    seg_start = boundary_line.interpolate(current_distance)
                    seg_end = boundary_line.interpolate(seg_end_distance)
                    
                    # Get visibility count for this segment
                    visibility_count = segment_visibility_counts.get(segment_idx, 0)
                    color = cmap(norm(visibility_count))
                    
                    # Line width based on visibility
                    line_width = 3 + 5 * (visibility_count / max_segment_visibility)
                    
                    # Extract the portion of boundary between start and end
                    coords = []
                    if seg_end_distance < boundary_length:
                        # Normal segment
                        num_samples = max(3, int((seg_end_distance - current_distance) / 0.5))
                        for i in range(num_samples + 1):
                            sample_distance = current_distance + i * (seg_end_distance - current_distance) / num_samples
                            sample_point = boundary_line.interpolate(sample_distance)
                            coords.append((sample_point.x, sample_point.y))
                    else:
                        # Last segment
                        remaining_distance = seg_end_distance - current_distance
                        num_samples = max(3, int(remaining_distance / 0.5))
                        for i in range(num_samples + 1):
                            sample_distance = current_distance + i * remaining_distance / num_samples
                            sample_point = boundary_line.interpolate(sample_distance)
                            coords.append((sample_point.x, sample_point.y))
                    
                    # Plot segment along boundary with color intensity
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    plt.plot(x_coords, y_coords, color=color, linewidth=line_width, alpha=0.8, zorder=3)
                    
                    # Plot segment endpoints with color
                    plt.plot(seg_start.x, seg_start.y, 'o', color=color, markersize=4, 
                            markeredgecolor='black', markeredgewidth=0.3, zorder=4)
                    plt.plot(seg_end.x, seg_end.y, 'o', color=color, markersize=4, 
                            markeredgecolor='black', markeredgewidth=0.3, zorder=4)
                    
                    # Move to next segment
                    current_distance += segment_size
                    segment_idx += 1
            
            # Create colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
            cbar.set_label('Number of Edges with Visibility', rotation=270, labelpad=20, fontsize=12)
        else:
            # No visibility data available
            for poly in building.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, 'g-', linewidth=2)
                plt.fill(x, y, alpha=0.3, color='green')
            
            # Plot segments in default color following boundary
            segment_size = self.config['visibility']['segment_size']
            boundary_lines = [poly.exterior for poly in building.geoms]
            
            segment_idx = 0
            for boundary_line in boundary_lines:
                current_distance = 0
                boundary_length = boundary_line.length
                
                while current_distance < boundary_length:
                    seg_end_distance = min(current_distance + segment_size, boundary_length)
                    seg_start = boundary_line.interpolate(current_distance)
                    seg_end = boundary_line.interpolate(seg_end_distance)
                    
                    # Extract boundary portion and plot
                    coords = []
                    if seg_end_distance < boundary_length:
                        num_samples = max(3, int((seg_end_distance - current_distance) / 0.5))
                        for i in range(num_samples + 1):
                            sample_distance = current_distance + i * (seg_end_distance - current_distance) / num_samples
                            sample_point = boundary_line.interpolate(sample_distance)
                            coords.append((sample_point.x, sample_point.y))
                    else:
                        remaining_distance = seg_end_distance - current_distance
                        num_samples = max(3, int(remaining_distance / 0.5))
                        for i in range(num_samples + 1):
                            sample_distance = current_distance + i * remaining_distance / num_samples
                            sample_point = boundary_line.interpolate(sample_distance)
                            coords.append((sample_point.x, sample_point.y))
                    
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)
                    plt.plot(seg_start.x, seg_start.y, 'ro', markersize=4, alpha=0.8)
                    plt.plot(seg_end.x, seg_end.y, 'ro', markersize=4, alpha=0.8)
                    
                    current_distance += segment_size
                    segment_idx += 1
            
            self.logger.warning("No segment visibility data available for heatmap")
        
        plt.title("Segment Visibility Heatmap\n(Number of Edges that Can See Each Segment)", 
                 fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='best')
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"2_segment_visibility_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['visibility_heatmaps']['format'], 
                   dpi=self.config['output']['visibility_heatmaps']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved segment visibility heatmap to {filename}")
        plt.close()
    
    def _plot_base_geometries(self, building, obstacles, show_legend=False):
        """
        Plot building and obstacles as base layer for heatmaps.
        """
        # Plot building
        for i, poly in enumerate(building.geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, 
                    label='Building' if i == 0 and show_legend else "")
            plt.fill(x, y, alpha=0.2, color='green')
        
        # Plot obstacles
        self._plot_obstacles_only(obstacles, show_legend)
    
    def _plot_obstacles_only(self, obstacles, show_legend=False):
        """
        Plot only obstacles without building.
        """
        # Plot radiation obstacles
        for i, poly in enumerate(obstacles['radiation'].geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.4, color='red', hatch='///', 
                    label='Radiation Obstacle' if i == 0 and show_legend else "")
        
        # Plot visibility-only obstacles
        visibility_only_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if not is_radiation:
                visibility_only_geoms.append(poly)
        
        for i, poly in enumerate(visibility_only_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2)
            plt.fill(x, y, alpha=0.4, color='blue', hatch='+++', 
                    label='Visibility Obstacle' if i == 0 and show_legend else "")
        
        # Plot combined obstacles
        combined_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if is_radiation:
                combined_geoms.append(poly)
        
        for i, poly in enumerate(combined_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'purple', linewidth=2)
            plt.fill(x, y, alpha=0.4, color='purple', hatch='xxx', 
                    label='Radiation + Visibility Obstacle' if i == 0 and show_legend else "")

    def create_thesis_plots(self, G, building, obstacles, segments, G_rotated=None, rotation_params=None):
        """
        Create thesis-specific plots for analysis.
        
        Args:
            G: Original networkx DiGraph
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            segments: List of segments
            G_rotated: Rotated networkx DiGraph (optional)
            rotation_params: Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle)
        """
        # Check if thesis plots are enabled
        if not self.config['output'].get('thesis_plots', {}).get('enabled', False):
            self.logger.info("Thesis plots are disabled in config")
            return
        
        self.logger.info("Creating thesis plots")
        
        # Create output directory
        thesis_plots_dir = self.config['output']['thesis_plots']['path']
        os.makedirs(thesis_plots_dir, exist_ok=True)
        
        # Plot 1: Building and obstacles footprint
        self._plot_building_obstacles_footprint(building, obstacles, thesis_plots_dir)
        
        # Plot 2: Building footprint with segments
        self._plot_building_with_segments(building, obstacles, segments, thesis_plots_dir)
        
        # Plot 3: Building with original graph network
        self._plot_building_with_original_graph(G, building, obstacles, thesis_plots_dir)
        
        # Plot 4: Building with rotated graph network
        if G_rotated is not None:
            self._plot_building_with_rotated_graph(G_rotated, building, obstacles, 
                                                 rotation_params, thesis_plots_dir)
    
    def _plot_building_obstacles_footprint(self, building, obstacles, output_dir):
        """
        Plot 1: Building and obstacles footprint with legend.
        """
        plt.figure(figsize=(12, 10))
        
        # Plot building
        for poly in building.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, label='Building' if 'Building' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.fill(x, y, alpha=0.3, color='green')
        
        # Plot radiation obstacles
        for i, poly in enumerate(obstacles['radiation'].geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='red', hatch='///', 
                    label='Floor Obstacle' if i == 0 else "")
        
        # Plot visibility obstacles
        visibility_only_geoms = []
        for poly in obstacles['visibility'].geoms:
            # Check if this polygon is also in radiation obstacles
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if not is_radiation:
                visibility_only_geoms.append(poly)
        
        for i, poly in enumerate(visibility_only_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='blue', hatch='+++', 
                    label='Wall Obstacle' if i == 0 else "")
        
        # Plot combined obstacles (both radiation and visibility)
        combined_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if is_radiation:
                combined_geoms.append(poly)
        
        for i, poly in enumerate(combined_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'purple', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='purple', hatch='xxx', 
                    label='Wall Obstacle' if i == 0 else "")
        
        plt.title("Building and Obstacles Footprint", fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"1_building_obstacles_footprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['thesis_plots']['format'], 
                   dpi=self.config['output']['thesis_plots']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved thesis plot 1 to {filename}")
        plt.close()
    
    def _plot_building_with_segments(self, building, obstacles, segments, output_dir):
        """
        Plot 2: Building footprint with segments following building boundary.
        """
        plt.figure(figsize=(14, 10))
        
        # Plot building
        for poly in building.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, label='Building' if 'Building' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.fill(x, y, alpha=0.3, color='green')
        
        # Plot obstacles (same as plot 1)
        for i, poly in enumerate(obstacles['radiation'].geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='red', hatch='///', 
                    label='Floor Obstacle' if i == 0 else "")
        
        visibility_only_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if not is_radiation:
                visibility_only_geoms.append(poly)
        
        for i, poly in enumerate(visibility_only_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='blue', hatch='+++', 
                    label='Wall Obstacle' if i == 0 else "")
        
        combined_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if is_radiation:
                combined_geoms.append(poly)
        
        for i, poly in enumerate(combined_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'purple', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='purple', hatch='xxx', 
                    label='Wall Obstacle' if i == 0 else "")
        
        # Plot segments following building boundary
        segment_size = self.config['visibility']['segment_size']
        
        # Get boundary lines from building polygons
        boundary_lines = [poly.exterior for poly in building.geoms]
        
        # Recreate segments properly along boundaries to match the original creation logic
        segment_idx = 0
        for boundary_line in boundary_lines:
            current_distance = 0
            boundary_length = boundary_line.length
            
            while current_distance < boundary_length:
                # Calculate segment end distance
                seg_end_distance = min(current_distance + segment_size, boundary_length)
                
                # Get start and end points on the boundary
                seg_start = boundary_line.interpolate(current_distance)
                seg_end = boundary_line.interpolate(seg_end_distance)
                
                # Extract the portion of boundary between start and end
                if seg_end_distance < boundary_length:
                    # Normal segment - extract line portion
                    coords = []
                    
                    # Sample points along the boundary segment for smooth curves
                    num_samples = max(3, int((seg_end_distance - current_distance) / 0.5))  # Sample every 0.5 units
                    for i in range(num_samples + 1):
                        sample_distance = current_distance + i * (seg_end_distance - current_distance) / num_samples
                        sample_point = boundary_line.interpolate(sample_distance)
                        coords.append((sample_point.x, sample_point.y))
                    
                    # Plot segment along boundary
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8,
                            label='Segments' if segment_idx == 0 else "")
                    
                    # Plot segment endpoints
                    plt.plot(seg_start.x, seg_start.y, 'ro', markersize=5, alpha=0.9)
                    plt.plot(seg_end.x, seg_end.y, 'ro', markersize=5, alpha=0.9)
                    
                else:
                    # Last segment or segment that reaches the end
                    coords = []
                    
                    # Sample points along the remaining boundary
                    remaining_distance = seg_end_distance - current_distance
                    num_samples = max(3, int(remaining_distance / 0.5))
                    for i in range(num_samples + 1):
                        sample_distance = current_distance + i * remaining_distance / num_samples
                        sample_point = boundary_line.interpolate(sample_distance)
                        coords.append((sample_point.x, sample_point.y))
                    
                    # Plot segment along boundary
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8,
                            label='Segments' if segment_idx == 0 else "")
                    
                    # Plot segment endpoints
                    plt.plot(seg_start.x, seg_start.y, 'ro', markersize=5, alpha=0.9)
                    plt.plot(seg_end.x, seg_end.y, 'ro', markersize=5, alpha=0.9)
                
                # Add segment ID labels
                midpoint_x = (seg_start.x + seg_end.x) / 2.0
                midpoint_y = (seg_start.y + seg_end.y) / 2.0
                plt.text(midpoint_x, midpoint_y, str(segment_idx), fontsize=7,
                        ha='center', va='center', color='darkred',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                
                # Move to next segment
                current_distance += segment_size
                segment_idx += 1
        
        plt.title(f"Segmentized Buidling Footprint (Segment Size: {segment_size}m)", 
                 fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='best')
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"2_building_with_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['thesis_plots']['format'], 
                   dpi=self.config['output']['thesis_plots']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved thesis plot 2 to {filename}")
        plt.close()
    
    def _plot_building_with_original_graph(self, G, building, obstacles, output_dir):
        """
        Plot 3: Building footprint with original graph network.
        """
        plt.figure(figsize=(14, 10))
        
        # Plot building
        for poly in building.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, label='Building' if 'Building' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.fill(x, y, alpha=0.3, color='green')
        
        # Plot obstacles
        for i, poly in enumerate(obstacles['radiation'].geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='red', hatch='///', 
                    label='Floor Obstacle' if i == 0 else "")
        
        visibility_only_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if not is_radiation:
                visibility_only_geoms.append(poly)
        
        for i, poly in enumerate(visibility_only_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='blue', hatch='+++', 
                    label='Wall Obstacle' if i == 0 else "")
        
        combined_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if is_radiation:
                combined_geoms.append(poly)
        
        for i, poly in enumerate(combined_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'purple', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='purple', hatch='xxx', 
                    label='Wall Obstacle' if i == 0 else "")
        
        # Plot graph network
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot edges
        nx.draw_networkx_edges(G, pos, width=1, edge_color='lightblue', alpha=0.6, arrows=False)
        
        # Plot nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='darkblue', alpha=0.8)
        
        grid_spacing = self.config['graph']['grid_spacing']
        plt.title(f"Building with Original Graph Network (Grid Spacing: {grid_spacing}m)", 
                 fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"3_building_original_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['thesis_plots']['format'], 
                   dpi=self.config['output']['thesis_plots']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved thesis plot 3 to {filename}")
        plt.close()
    
    def _plot_building_with_rotated_graph(self, G_rotated, building, obstacles, rotation_params, output_dir):
        """
        Plot 4: Building footprint with rotated graph network.
        """
        plt.figure(figsize=(14, 10))
        
        # Plot building
        for poly in building.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'g-', linewidth=2, label='Building' if 'Building' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.fill(x, y, alpha=0.3, color='green')
        
        # Plot obstacles
        for i, poly in enumerate(obstacles['radiation'].geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='red', hatch='///', 
                    label='Floor Obstacle' if i == 0 else "")
        
        visibility_only_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if not is_radiation:
                visibility_only_geoms.append(poly)
        
        for i, poly in enumerate(visibility_only_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='blue', hatch='+++', 
                    label='Wall Obstacle' if i == 0 else "")
        
        combined_geoms = []
        for poly in obstacles['visibility'].geoms:
            is_radiation = any(poly.equals(rad_poly) for rad_poly in obstacles['radiation'].geoms)
            if is_radiation:
                combined_geoms.append(poly)
        
        for i, poly in enumerate(combined_geoms):
            x, y = poly.exterior.xy
            plt.plot(x, y, 'purple', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='purple', hatch='xxx', 
                    label='Wall Obstacle' if i == 0 else "")
        
        # Plot rotated graph network
        pos = nx.get_node_attributes(G_rotated, 'pos')
        
        # Plot edges
        nx.draw_networkx_edges(G_rotated, pos, width=1, edge_color='orange', alpha=0.6, arrows=False)
        
        # Plot nodes
        nx.draw_networkx_nodes(G_rotated, pos, node_size=20, node_color='darkorange', alpha=0.8)
        
        # Add rotation information to title
        title = f"Building with Rotated Graph Network"
        if rotation_params:
            rotation_angle, rotation_center, longest_edge_angle, target_angle = rotation_params
            title += f" (Rotated {rotation_angle:.1f}° to align with {target_angle}°)"
        
        grid_spacing = self.config['graph']['grid_spacing']
        title += f"\nGrid Spacing: {grid_spacing}m"
        
        plt.title(title, fontsize=12)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.axis('equal')
        
        # Save plot
        filename = os.path.join(output_dir, f"4_building_rotated_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, format=self.config['output']['thesis_plots']['format'], 
                   dpi=self.config['output']['thesis_plots']['dpi'], bbox_inches='tight')
        self.logger.info(f"Saved thesis plot 4 to {filename}")
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
        for poly in obstacles['radiation'].geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            plt.fill(x, y, alpha=0.6, color='red', hatch='///')
        
        # Plot visibility obstacles
        for poly in obstacles['visibility'].geoms:
            if poly not in obstacles['radiation'].geoms:  # Avoid plotting twice
                x, y = poly.exterior.xy
                plt.plot(x, y, 'k-', linewidth=2)
                plt.fill(x, y, alpha=0.6, color='black')
    
    def _plot_segments_on_building(self, building, segments):
        """
        Plot the building segments following the building boundary.
        
        Args:
            building: MultiPolygon representing the building
            segments: List of segments
        """
        # Plot segments following building boundary
        segment_size = self.config['visibility']['segment_size']
        
        # Get boundary lines from building polygons
        boundary_lines = [poly.exterior for poly in building.geoms]
        
        # Recreate segments properly along boundaries to match the original creation logic
        segment_idx = 0
        for boundary_line in boundary_lines:
            current_distance = 0
            boundary_length = boundary_line.length
            
            while current_distance < boundary_length:
                # Calculate segment end distance
                seg_end_distance = min(current_distance + segment_size, boundary_length)
                
                # Get start and end points on the boundary
                seg_start = boundary_line.interpolate(current_distance)
                seg_end = boundary_line.interpolate(seg_end_distance)
                
                # Extract the portion of boundary between start and end
                coords = []
                if seg_end_distance < boundary_length:
                    # Normal segment - extract line portion
                    num_samples = max(3, int((seg_end_distance - current_distance) / 0.5))  # Sample every 0.5 units
                    for i in range(num_samples + 1):
                        sample_distance = current_distance + i * (seg_end_distance - current_distance) / num_samples
                        sample_point = boundary_line.interpolate(sample_distance)
                        coords.append((sample_point.x, sample_point.y))
                else:
                    # Last segment or segment that reaches the end
                    remaining_distance = seg_end_distance - current_distance
                    num_samples = max(3, int(remaining_distance / 0.5))
                    for i in range(num_samples + 1):
                        sample_distance = current_distance + i * remaining_distance / num_samples
                        sample_point = boundary_line.interpolate(sample_distance)
                        coords.append((sample_point.x, sample_point.y))
                
                # Plot segment along boundary
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)
                
                # Plot segment endpoints
                plt.plot(seg_start.x, seg_start.y, 'ro', markersize=3, alpha=0.9)
                plt.plot(seg_end.x, seg_end.y, 'ro', markersize=3, alpha=0.9)
                
                # Add segment ID labels if enabled
                if self.config['visualization']['show_segment_ids']:
                    midpoint_x = (seg_start.x + seg_end.x) / 2.0
                    midpoint_y = (seg_start.y + seg_end.y) / 2.0
                    plt.text(
                        midpoint_x, midpoint_y,
                        str(segment_idx),
                        fontsize=8,
                        ha='center',
                        va='center',
                        color='blue',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                    )
                
                # Move to next segment
                current_distance += segment_size
                segment_idx += 1
    
    def _plot_segments(self, segments):
        """
        Plot the building segments following the building boundary.
        This is a fallback method when building geometry is not available.
        
        Args:
            segments: List of segments
        """
        # This method needs access to building geometry to plot segments correctly
        # For now, plot segment endpoints only since we don't have building reference here
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
        
    def highlight_alignment(self, rotation_params):
        """
        Add alignment indicators to the plot.
        
        Args:
            rotation_params: Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle)
        """
        if not rotation_params:
            return
        
        rotation_angle, rotation_center, longest_edge_angle, target_angle = rotation_params
        
        # Calculate bounds of the plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Calculate diagonal length
        diag_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) * 0.2
        
        # Draw north direction
        plt.arrow(
            rotation_center[0], rotation_center[1],
            0, diag_length * 0.1,
            head_width=diag_length * 0.02,
            head_length=diag_length * 0.03,
            fc='black',
            ec='black',
            zorder=10,
            label='North'
        )
        
        # Draw original longest edge direction
        dx = diag_length * 0.1 * np.sin(np.radians(longest_edge_angle))
        dy = diag_length * 0.1 * np.cos(np.radians(longest_edge_angle))
        plt.arrow(
            rotation_center[0], rotation_center[1],
            dx, dy,
            head_width=diag_length * 0.02,
            head_length=diag_length * 0.03,
            fc='red',
            ec='red',
            zorder=10,
            label=f'Longest Edge ({longest_edge_angle:.1f}°)'
        )
        
        # Draw target alignment direction
        dx = diag_length * 0.1 * np.sin(np.radians(target_angle))
        dy = diag_length * 0.1 * np.cos(np.radians(target_angle))
        plt.arrow(
            rotation_center[0], rotation_center[1],
            dx, dy,
            head_width=diag_length * 0.02,
            head_length=diag_length * 0.03,
            fc='green',
            ec='green',
            zorder=10,
            label=f'Grid Alignment ({target_angle}°)'
        )
        
        plt.legend(loc='upper right')