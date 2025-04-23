"""
Data handler for loading and processing geometry data.
"""
import json
import logging
from shapely.geometry import Polygon, MultiPolygon

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