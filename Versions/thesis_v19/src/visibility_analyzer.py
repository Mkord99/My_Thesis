"""
Visibility analyzer for calculating visibility between edges and building segments.
Parallelized version that utilizes multiple CPU cores.
"""
import logging
import numpy as np
import multiprocessing
import os
from functools import partial
from itertools import islice
from shapely.geometry import Point, LineString
from src.utils import (
    calculate_angle, 
    log_memory_usage,
    calculate_normal_vector,
    is_within_angle_constraint,
    save_visibility_data
)

class VisibilityAnalyzer:
    """Analyzes visibility between edges and building segments."""
    
    def __init__(self, config):
        """
        Initialize the visibility analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set number of CPU cores to use
        # If num_cores is None or not provided, use all available cores
        cores_in_config = config.get('performance', {}).get('num_cores')
        if cores_in_config is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = int(cores_in_config)
        
        # Set touch threshold for visibility checks
        self.touch_threshold = 0.1  # Small distance threshold for touch checks
            
        self.logger.info(f"Using {self.num_cores} CPU cores for visibility analysis")
        self.logger.info(f"Using touch threshold of {self.touch_threshold} meters")
    
    def analyze(self, G, grid_points, building, obstacles):
        """
        Analyze visibility between edges and building segments.
        
        Args:
            G: networkx DiGraph
            grid_points: List of grid points
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            Tuple of (segments, segment_visibility, edge_visibility, vrf)
        """
        self.logger.info("Analyzing visibility")
        log_memory_usage(self.logger, "Before segment creation")
        
        # Create segments along the building boundaries
        segments, segment_normals = self._create_segments_with_normals(building)
        self.logger.info(f"Created {len(segments)} building segments with normal vectors")
        
        # Calculate edge-segment visibility in parallel
        self.logger.info("Calculating edge-segment visibility using normal vector approach (parallelized)")
        log_memory_usage(self.logger, "Before edge-segment visibility calculation")
        edge_visibility, segment_visibility = self._calculate_edge_segment_visibility_parallel(
            G, grid_points, segments, segment_normals, building, obstacles
        )
        log_memory_usage(self.logger, "After edge-segment visibility calculation")
        
        # Calculate particle-based visibility if enabled and use those results
        # Otherwise, use the direct edge-segment visibility results
        if self.config['visibility']['particle_visibility']['enabled']:
            self.logger.info("Calculating particle-based visibility using normal vector approach (parallelized)")
            log_memory_usage(self.logger, "Before particle visibility calculation")
            edge_particle_visibility = self._calculate_particle_visibility_parallel(
                G, segments, segment_normals, building, obstacles
            )
            
            # Update segment visibility based on particles
            segment_visibility_particles = self._update_segment_visibility(
                segments, edge_particle_visibility
            )
            
            # Calculate Visibility Ratio Factor (VRF)
            vrf = self._calculate_vrf(G, edge_particle_visibility)
            log_memory_usage(self.logger, "After particle visibility calculation")
            
            # Use the particle-based visibility results
            final_segment_visibility = segment_visibility_particles
            final_edge_visibility = self._update_edge_visibility(edge_visibility, edge_particle_visibility)
        else:
            # Use the direct edge-segment visibility results
            final_segment_visibility = segment_visibility
            final_edge_visibility = edge_visibility
            vrf = {edge: 1.0 for edge in G.edges()}
            
        # Save visibility data to files (after all visibility calculations are complete)
        self._save_visibility_data(final_segment_visibility, final_edge_visibility)
        
        return segments, final_segment_visibility, final_edge_visibility, vrf
    
    def _create_segments_with_normals(self, building):
        """
        Create segments along the building boundaries and compute their normal vectors.
        
        Args:
            building: MultiPolygon representing the building
            
        Returns:
            Tuple of (segments list, normal vectors dictionary)
        """
        segment_size = self.config['visibility']['segment_size']
        
        # Get boundary lines from building polygons
        boundary_lines = [poly.exterior for poly in building.geoms]
        
        # Create segments along each boundary line
        segments = []
        segment_normals = {}
        
        for boundary_line in boundary_lines:
            for i in range(0, int(boundary_line.length), segment_size):
                seg_start = boundary_line.interpolate(i)
                seg_end = boundary_line.interpolate(min(i + segment_size, boundary_line.length))
                
                # Create segment
                segment = (seg_start, seg_end)
                seg_idx = len(segments)
                segments.append(segment)
                
                # Calculate segment vector
                segment_vec = (seg_end.x - seg_start.x, seg_end.y - seg_start.y)
                
                # Calculate segment midpoint
                segment_midpoint = Point((seg_start.x + seg_end.x) / 2, (seg_start.y + seg_end.y) / 2)
                
                # Calculate normal vector pointing outward
                normal_vec = calculate_normal_vector(segment_vec, building, segment_midpoint)
                segment_normals[seg_idx] = normal_vec
        
        return segments, segment_normals
    
    def _chunk_data(self, data, n_chunks):
        """
        Split data into approximately equal sized chunks.
        
        Args:
            data: Data to split
            n_chunks: Number of chunks
            
        Returns:
            List of chunks
        """
        # Ensure data is a list (in case it's an iterator)
        data_list = list(data)
        if not data_list:
            return []
            
        k, m = divmod(len(data_list), n_chunks)
        return [list(islice(data_list, i * k + min(i, m), (i + 1) * k + min(i + 1, m))) for i in range(n_chunks)]
    
    def _process_segment_chunk(self, seg_chunk, G, grid_points, segment_normals, building, obstacles, vis_config):
        """
        Process a chunk of segments to calculate visibility using normal vector approach.
        
        Args:
            seg_chunk: List of (segment_index, segment) tuples
            G: networkx DiGraph
            grid_points: List of grid points
            segment_normals: Dictionary mapping segment indices to normal vectors
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            vis_config: Dictionary of visibility constraints
            
        Returns:
            Tuple of (segment_visibility, edge_visibility)
        """
        local_segment_visibility = {}
        local_edge_visibility = {edge: [] for edge in G.edges()}
        
        min_distance = vis_config['min_distance']
        max_distance = vis_config['max_distance']
        max_normal_angle = vis_config['max_normal_angle']
        
        for seg_idx, (seg_start, seg_end) in seg_chunk:
            # Get the normal vector for this segment
            normal_vec = segment_normals[seg_idx]
            
            local_segment_visibility[seg_idx] = []
            
            # Calculate segment midpoint
            segment_midpoint = Point((seg_start.x + seg_end.x) / 2, (seg_start.y + seg_end.y) / 2)
            
            for edge in G.edges():
                p1_idx, p2_idx = edge
                p1, p2 = grid_points[p1_idx], grid_points[p2_idx]
                
                # Calculate vectors from segment midpoint to edge endpoints
                to_p1_vec = (p1.x - segment_midpoint.x, p1.y - segment_midpoint.y)
                to_p2_vec = (p2.x - segment_midpoint.x, p2.y - segment_midpoint.y)
                
                # Calculate distances to segment endpoints
                d1_start = p1.distance(seg_start)
                d1_end = p1.distance(seg_end)
                d2_start = p2.distance(seg_start)
                d2_end = p2.distance(seg_end)
                
                # Check distance constraints
                if not ((min_distance <= d1_start <= max_distance and min_distance <= d1_end <= max_distance) or
                        (min_distance <= d2_start <= max_distance and min_distance <= d2_end <= max_distance)):
                    continue
                
                # Create sight lines
                line1_start = LineString([p1, seg_start])
                line1_end = LineString([p1, seg_end])
                line2_start = LineString([p2, seg_start])
                line2_end = LineString([p2, seg_end])
                
                # Check if sight lines touch the building (required for visibility)
                # IMPORTANT: Use distance threshold instead of exact touch check
                touches1_start = line1_start.distance(building) < self.touch_threshold
                touches1_end = line1_end.distance(building) < self.touch_threshold
                touches2_start = line2_start.distance(building) < self.touch_threshold
                touches2_end = line2_end.distance(building) < self.touch_threshold
                
                # Skip if both points' visibility is blocked by obstacles
                if ((line1_start.intersects(obstacles['visibility']) or line1_end.intersects(obstacles['visibility'])) and
                    (line2_start.intersects(obstacles['visibility']) or line2_end.intersects(obstacles['visibility']))):
                    continue
                
                # Check visibility conditions for either endpoint using normal vector approach
                is_visible = False
                
                # Check first point visibility with normal vector
                if (min_distance <= d1_start <= max_distance and 
                    min_distance <= d1_end <= max_distance and 
                    is_within_angle_constraint(normal_vec, to_p1_vec, max_normal_angle) and
                    touches1_start and touches1_end):
                    is_visible = True
                
                # Check second point visibility with normal vector
                if (min_distance <= d2_start <= max_distance and 
                    min_distance <= d2_end <= max_distance and 
                    is_within_angle_constraint(normal_vec, to_p2_vec, max_normal_angle) and
                    touches2_start and touches2_end):
                    is_visible = True
                
                # Update visibility records if visible
                if is_visible:
                    local_segment_visibility[seg_idx].append(edge)
                    local_edge_visibility[edge].append(seg_idx)
        
        return local_segment_visibility, local_edge_visibility

    def _calculate_edge_segment_visibility_parallel(self, G, grid_points, segments, segment_normals, building, obstacles):
        """
        Calculate visibility between edges and segments using parallel processing.
        
        Args:
            G: networkx DiGraph
            grid_points: List of grid points
            segments: List of segments
            segment_normals: Dictionary mapping segment indices to normal vectors
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            Tuple of (edge_visibility, segment_visibility)
        """
        # Create segment chunks for parallel processing
        if not segments:
            return {}, {}
            
        n_chunks = min(self.num_cores, len(segments))
        if n_chunks <= 0:
            n_chunks = 1  # Ensure at least one chunk
            
        segment_indices = [(i, segment) for i, segment in enumerate(segments)]
        segment_chunks = self._chunk_data(segment_indices, n_chunks)
        
        # Prepare visibility constraints
        vis_config = self.config['visibility']['visibility_constraints']
        
        self.logger.info(f"Dividing {len(segments)} segments into {n_chunks} chunks for parallel processing")
        
        # Create a pool of worker processes
        pool = None
        try:
            pool = multiprocessing.Pool(processes=self.num_cores)
            
            # Process each chunk in parallel
            results = pool.map(
                partial(
                    self._process_segment_chunk, 
                    G=G, 
                    grid_points=grid_points, 
                    segment_normals=segment_normals,
                    building=building, 
                    obstacles=obstacles,
                    vis_config=vis_config
                ), 
                segment_chunks
            )
            
            # Combine results
            segment_visibility = {}
            edge_visibility = {edge: [] for edge in G.edges()}
            
            for local_segment_vis, local_edge_vis in results:
                # Merge segment visibility
                for seg_idx, edges in local_segment_vis.items():
                    if seg_idx not in segment_visibility:
                        segment_visibility[seg_idx] = []
                    segment_visibility[seg_idx].extend(edges)
                
                # Merge edge visibility
                for edge, seg_indices in local_edge_vis.items():
                    edge_visibility[edge].extend(seg_indices)
            
            return edge_visibility, segment_visibility
            
        finally:
            # Ensure pool is properly closed
            if pool is not None:
                pool.close()
                pool.join()
    
    def _process_edge_chunk(self, edge_chunk, G, segments, segment_normals, building, obstacles, particle_spacing, vis_config):
        """
        Process a chunk of edges to calculate particle visibility using normal vector approach.
        
        Args:
            edge_chunk: List of edges
            G: networkx DiGraph
            segments: List of segments
            segment_normals: Dictionary mapping segment indices to normal vectors
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            particle_spacing: Spacing between particles
            vis_config: Dictionary of visibility constraints
            
        Returns:
            Dictionary mapping edges to particle visibility
        """
        local_edge_particle_visibility = {}
        
        min_distance = vis_config['min_distance']
        max_distance = vis_config['max_distance']
        max_normal_angle = vis_config['max_normal_angle']
        
        for edge in edge_chunk:
            p1_idx, p2_idx = edge
            p1 = Point(G.nodes[p1_idx]['pos'])
            p2 = Point(G.nodes[p2_idx]['pos'])
            
            # Create a line for the edge
            edge_line = LineString([p1, p2])
            edge_length = edge_line.length
            
            # Sample points along the edge
            sample_points = [edge_line.interpolate(d) for d in np.arange(0, edge_length + 1e-6, particle_spacing)]
            
            local_edge_particle_visibility[edge] = {}
            
            # Check visibility for each particle (adjacent sample points)
            for idx in range(len(sample_points) - 1):
                part_start = sample_points[idx]
                part_end = sample_points[idx+1]
                particle_vis = []
                
                # Check visibility to each segment
                for seg_idx, (seg_start, seg_end) in enumerate(segments):
                    # Get normal vector for this segment
                    normal_vec = segment_normals[seg_idx]
                    
                    # Calculate segment midpoint
                    segment_midpoint = Point((seg_start.x + seg_end.x) / 2, (seg_start.y + seg_end.y) / 2)
                    
                    # Calculate vectors from segment midpoint to particles
                    to_part_start_vec = (part_start.x - segment_midpoint.x, part_start.y - segment_midpoint.y)
                    to_part_end_vec = (part_end.x - segment_midpoint.x, part_end.y - segment_midpoint.y)
                    
                    # Calculate distances
                    d1_start = part_start.distance(seg_start)
                    d1_end = part_start.distance(seg_end)
                    d2_start = part_end.distance(seg_start)
                    d2_end = part_end.distance(seg_end)
                    
                    # Check distance constraints
                    if not ((min_distance <= d1_start <= max_distance and min_distance <= d1_end <= max_distance) or
                            (min_distance <= d2_start <= max_distance and min_distance <= d2_end <= max_distance)):
                        continue
                    
                    # Create sight lines
                    line1_start = LineString([part_start, seg_start])
                    line1_end = LineString([part_start, seg_end])
                    line2_start = LineString([part_end, seg_start])
                    line2_end = LineString([part_end, seg_end])
                    
                    # Check if sight lines touch the building
                    # IMPORTANT: Use distance threshold instead of exact touch check
                    touches1_start = line1_start.distance(building) < self.touch_threshold
                    touches1_end = line1_end.distance(building) < self.touch_threshold
                    touches2_start = line2_start.distance(building) < self.touch_threshold
                    touches2_end = line2_end.distance(building) < self.touch_threshold
                    
                    # Skip if both particles' visibility is blocked by obstacles
                    if ((line1_start.intersects(obstacles['visibility']) or line1_end.intersects(obstacles['visibility'])) and
                        (line2_start.intersects(obstacles['visibility']) or line2_end.intersects(obstacles['visibility']))):
                        continue
                    
                    # Check visibility conditions using normal vector approach
                    if ((min_distance <= d1_start <= max_distance and 
                         min_distance <= d1_end <= max_distance and 
                         is_within_angle_constraint(normal_vec, to_part_start_vec, max_normal_angle) and
                         touches1_start and touches1_end) or
                        (min_distance <= d2_start <= max_distance and 
                         min_distance <= d2_end <= max_distance and 
                         is_within_angle_constraint(normal_vec, to_part_end_vec, max_normal_angle) and
                         touches2_start and touches2_end)):
                        particle_vis.append(seg_idx)
                
                local_edge_particle_visibility[edge][idx] = particle_vis
        
        return local_edge_particle_visibility

    def _calculate_particle_visibility_parallel(self, G, segments, segment_normals, building, obstacles):
        """
        Calculate particle-based visibility along edges using parallel processing.
        
        Args:
            G: networkx DiGraph
            segments: List of segments
            segment_normals: Dictionary mapping segment indices to normal vectors
            building: MultiPolygon representing the building
            obstacles: Dictionary containing MultiPolygons for different obstacle types
            
        Returns:
            Dictionary mapping edges to particle visibility
        """
        # Create edge chunks for parallel processing
        edge_list = list(G.edges())
        if not edge_list:
            return {}
            
        n_chunks = min(self.num_cores, len(edge_list))
        if n_chunks <= 0:
            n_chunks = 1  # Ensure at least one chunk
            
        edge_chunks = self._chunk_data(edge_list, n_chunks)
        
        # Prepare config values
        particle_spacing = self.config['visibility']['particle_visibility']['spacing']
        vis_config = self.config['visibility']['visibility_constraints']
        
        self.logger.info(f"Dividing {len(edge_list)} edges into {n_chunks} chunks for parallel processing")
        
        # Create a pool of worker processes
        pool = None
        try:
            pool = multiprocessing.Pool(processes=self.num_cores)
            
            # Process each chunk in parallel
            results = pool.map(
                partial(
                    self._process_edge_chunk, 
                    G=G, 
                    segments=segments, 
                    segment_normals=segment_normals,
                    building=building, 
                    obstacles=obstacles,
                    particle_spacing=particle_spacing,
                    vis_config=vis_config
                ), 
                edge_chunks
            )
            
            # Combine results
            edge_particle_visibility = {}
            for local_result in results:
                edge_particle_visibility.update(local_result)
            
            return edge_particle_visibility
            
        finally:
            # Ensure pool is properly closed
            if pool is not None:
                pool.close()
                pool.join()
    
    def _update_segment_visibility(self, segments, edge_particle_visibility):
        """
        Update segment visibility based on particle visibility.
        
        Args:
            segments: List of segments
            edge_particle_visibility: Dictionary of particle visibility
            
        Returns:
            Updated segment visibility dictionary
        """
        segment_visibility_particles = {}
        
        # Initialize segment visibility
        for seg_idx in range(len(segments)):
            segment_visibility_particles[seg_idx] = []
            
            # Check each edge
            for edge, particles in edge_particle_visibility.items():
                # Check if any particle on this edge can see the segment
                for part_idx, particle_vis in particles.items():
                    if seg_idx in particle_vis:
                        segment_visibility_particles[seg_idx].append(edge)
                        break  # One particle is enough
        
        return segment_visibility_particles
    
    def _update_edge_visibility(self, edge_visibility, edge_particle_visibility):
        """
        Update edge visibility based on particle visibility.
        
        Args:
            edge_visibility: Original edge visibility dictionary
            edge_particle_visibility: Dictionary of particle visibility
            
        Returns:
            Updated edge visibility dictionary
        """
        updated_edge_visibility = {edge: [] for edge in edge_visibility}
        
        # For each edge, combine visible segments from original check and particle check
        for edge in edge_visibility:
            visible_segments = set(edge_visibility[edge])
            
            # Add segments visible from particles
            if edge in edge_particle_visibility:
                for part_idx, particle_vis in edge_particle_visibility[edge].items():
                    visible_segments.update(particle_vis)
            
            updated_edge_visibility[edge] = list(visible_segments)
        
        return updated_edge_visibility
    
    def _calculate_vrf(self, G, edge_particle_visibility):
        """
        Calculate Visibility Ratio Factor (VRF) for each edge.
        
        Args:
            G: networkx DiGraph
            edge_particle_visibility: Dictionary of particle visibility
            
        Returns:
            Dictionary of VRF values
        """
        epsilon = self.config['optimization']['epsilon']
        vrf = {}
        
        for edge in G.edges():
            p1_idx, p2_idx = edge
            p1 = Point(G.nodes[p1_idx]['pos'])
            p2 = Point(G.nodes[p2_idx]['pos'])
            
            edge_line = LineString([p1, p2])
            edge_length = edge_line.length
            
            # Combine visible segments from all particles
            visible_segments = set()
            for part_idx in edge_particle_visibility[edge]:
                visible_segments.update(edge_particle_visibility[edge][part_idx])
            
            # Calculate VRF
            vrf[edge] = len(visible_segments) / (edge_length + epsilon)
        
        return vrf
    
    def _save_visibility_data(self, segment_visibility, edge_visibility):
        """
        Save visibility data to CSV files.
        
        Args:
            segment_visibility: Dictionary mapping segments to edges that can see them
            edge_visibility: Dictionary mapping edges to segments they can see
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join("output", "visibility")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save segment visibility
        segment_vis_file = os.path.join(output_dir, "segment_visibility.csv")
        save_visibility_data(segment_vis_file, segment_visibility, is_edge_visibility=False)
        self.logger.info(f"Saved segment visibility to {segment_vis_file}")
        
        # Save edge visibility
        edge_vis_file = os.path.join(output_dir, "edge_visibility.csv")
        save_visibility_data(edge_vis_file, edge_visibility, is_edge_visibility=True)
        self.logger.info(f"Saved edge visibility to {edge_vis_file}")