"""
Data handler for loading and processing geometry data.
"""
import json
import logging
import os
from shapely.geometry import Polygon, MultiPolygon
from src.rotation_utils import (
    process_building_orientation, 
    rotate_geometries,
    save_debug_visualizations
)

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
        self.original_building = None
        self.original_obstacles = None
        self.rotation_enabled = config.get('rotation', {}).get('enabled', False)
        # Extract debug setting from config
        self.debug_visualization = config.get('rotation', {}).get('debug_visualization', False)
        self.debug_info = {}
    
    def load_geometries(self):
        """
        Load building and obstacle geometries from the specified file.
        Applies rotation preprocessing if enabled in configuration.
            
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
        
        # Store original geometries
        self.original_building = building
        self.original_obstacles = obstacles
        
        # Apply rotation preprocessing if enabled
        if self.rotation_enabled:
            self.logger.info("Rotation preprocessing is enabled")
            self.logger.info("Analyzing building orientation and applying rotation preprocessing")
            
            # Use enhanced process_building_orientation with debug visualization
            rotation_results = process_building_orientation(
                building, 
                debug_visualization=self.debug_visualization
            )
            
            self.rotation_angle, self.rotation_center, longest_edge_angle, target_angle, debug_info = rotation_results
            self.debug_info.update(debug_info)
            
            # Save orientation information to a file
            self._save_orientation_info(longest_edge_angle, target_angle, self.rotation_angle)
            
            # Rotate building and obstacles with debug visualization
            building, obstacles, rotate_debug_info = rotate_geometries(
                building, 
                obstacles, 
                self.rotation_angle, 
                self.rotation_center,
                debug_visualization=self.debug_visualization
            )
            
            self.debug_info.update(rotate_debug_info)
            
            # Save debug visualizations if enabled
            if self.debug_visualization:
                save_debug_visualizations(self.debug_info)
                
            self.logger.info(f"Rotated building and obstacles by {self.rotation_angle:.2f} degrees")
            
            # Log verification results if available
            if 'verification' in self.debug_info:
                verify = self.debug_info['verification']
                self.logger.info(f"Rotation Verification - Original: {verify['original_angle']:.2f}°, "
                               f"Rotated: {verify['rotated_angle']:.2f}°, "
                               f"Target: {verify['target_angle']}°, "
                               f"Error: {verify['error']:.4f}°")
        else:
            self.logger.info("Rotation preprocessing is disabled")
        
        # Return the building, obstacles and all polygons
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
            Tuple of (rotation_angle, rotation_center, rotation_enabled)
        """
        return self.rotation_angle, self.rotation_center, self.rotation_enabled
    
    def get_original_geometries(self):
        """
        Get original (unrotated) geometries.
        
        Returns:
            Tuple of (original_building, original_obstacles)
        """
        return self.original_building, self.original_obstacles
    
    def get_debug_info(self):
        """
        Get debug information.
        
        Returns:
            Debug information dictionary
        """
        return self.debug_info
    
    def _save_orientation_info(self, longest_edge_angle, target_angle, rotation_angle):
        """
        Save building orientation information to a file.
        
        Args:
            longest_edge_angle: Angle of the longest edge with north
            target_angle: Target angle for alignment
            rotation_angle: Applied rotation angle
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join("output", "orientation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save orientation information
        info = {
            "longest_edge_angle": longest_edge_angle,
            "target_angle": target_angle,
            "rotation_angle": rotation_angle,
            "rotation_center": self.rotation_center
        }
        
        # Add verification data if available
        if 'verification' in self.debug_info:
            info['verification'] = self.debug_info['verification']
        
        # Determine orientation classification
        if target_angle == 0:
            orientation = "Vertical (North)"
        elif target_angle == 90:
            orientation = "Horizontal (East)"
        else:  # target_angle == 180
            orientation = "Vertical (South)"
        
        info["orientation"] = orientation
        
        # Save to file
        with open(os.path.join(output_dir, "orientation_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Saved orientation information to {os.path.join(output_dir, 'orientation_info.json')}")