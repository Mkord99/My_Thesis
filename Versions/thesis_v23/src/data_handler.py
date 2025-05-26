"""
Data handler for loading and processing geometry data.
"""
import json
import logging
import os
from shapely.geometry import Polygon, MultiPolygon
from src.rotation_utils import process_building_orientation

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
        self.rotation_angle = 0
        self.rotation_center = (0, 0)
        self.longest_edge_angle = 0
        self.target_angle = 0
        self.rotation_enabled = config.get('rotation', {}).get('enabled', False)
        # Extract debug setting from config
        self.debug_visualization = config.get('rotation', {}).get('debug_visualization', False)
        self.debug_info = {}
    
    def load_geometries(self):
        """
        Load building and obstacle geometries from the specified file.
        Calculates rotation parameters if rotation is enabled, but does NOT rotate the geometries.
            
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
                    
                if 'buildings' not in all_polygons:
                    all_polygons['buildings'] = []
                if 'radiation_obstacles' not in all_polygons:
                    all_polygons['radiation_obstacles'] = []
                if 'visibility_obstacles' not in all_polygons:
                    all_polygons['visibility_obstacles'] = []
                
                all_polygons['radiation_obstacles'].extend(radiation_obstacles)
                all_polygons['visibility_obstacles'].extend(visibility_obstacles)
                
            except Exception as e:
                self.logger.error(f"Error processing obstacle {obstacle.get('id', 'unknown')}: {e}")
                raise
        
        # Create MultiPolygons for obstacles
        radiation_obstacles_multi = MultiPolygon(radiation_obstacles) if radiation_obstacles else MultiPolygon([])
        visibility_obstacles_multi = MultiPolygon(visibility_obstacles) if visibility_obstacles else MultiPolygon([])
        
        obstacles = {
            'radiation': radiation_obstacles_multi,
            'visibility': visibility_obstacles_multi
        }
        
        # Calculate rotation parameters if enabled, but DON'T rotate geometries
        if self.rotation_enabled:
            self.logger.info("Building orientation analysis is enabled")
            
            # Use enhanced process_building_orientation with debug visualization
            rotation_results = process_building_orientation(
                building, 
                debug_visualization=self.debug_visualization
            )
            
            self.rotation_angle, self.rotation_center, self.longest_edge_angle, self.target_angle, debug_info = rotation_results
            self.debug_info.update(debug_info)
            
            # Save orientation information to a file
            self._save_orientation_info()
            
            # Instead of rotating the buildings, we'll use the angle to create an aligned grid
            self.logger.info(f"Building orientation: Longest edge at {self.longest_edge_angle:.2f}°, "
                           f"Target angle {self.target_angle}°")
            self.logger.info(f"Grid alignment: Will use rotation angle of {self.rotation_angle:.2f}° "
                           f"around {self.rotation_center}")
            
            # Save debug visualizations if enabled
            if self.debug_visualization and 'before_rotation_fig' in self.debug_info:
                self._save_debug_visualizations()
                
        else:
            self.logger.info("Building orientation analysis is disabled")
        
        # Return the building, obstacles and all polygons (unrotated)
        return building, obstacles, all_polygons
    
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
    
    def get_rotation_params(self):
        """
        Get rotation parameters.
        
        Returns:
            Tuple of (rotation_angle, rotation_center, longest_edge_angle, target_angle, rotation_enabled)
        """
        return (self.rotation_angle, self.rotation_center, 
                self.longest_edge_angle, self.target_angle, self.rotation_enabled)
    
    def get_debug_info(self):
        """
        Get debug information.
        
        Returns:
            Debug information dictionary
        """
        return self.debug_info
    
    def _save_orientation_info(self):
        """
        Save building orientation information to a file.
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join("output", "orientation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save orientation information
        info = {
            "longest_edge_angle": self.longest_edge_angle,
            "target_angle": self.target_angle,
            "rotation_angle": self.rotation_angle,
            "rotation_center": self.rotation_center
        }
        
        # Determine orientation classification
        if self.target_angle == 0:
            orientation = "Vertical (North)"
        elif self.target_angle == 90:
            orientation = "Horizontal (East)"
        else:  # target_angle == 180
            orientation = "Vertical (South)"
        
        info["orientation"] = orientation
        info["grid_alignment"] = True
        
        # Save to file
        with open(os.path.join(output_dir, "orientation_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Saved orientation information to {os.path.join(output_dir, 'orientation_info.json')}")
    
    def _save_debug_visualizations(self):
        """
        Save debug visualizations to files.
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join("output", "orientation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save before rotation figure if available
        if 'before_rotation_fig' in self.debug_info:
            before_path = os.path.join(output_dir, "building_orientation.png")
            self.debug_info['before_rotation_fig'].savefig(before_path, dpi=300, bbox_inches='tight')
            print(f"Saved building orientation visualization to {before_path}")