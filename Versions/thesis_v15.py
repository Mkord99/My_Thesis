# Modulirized Version by Claude

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from scipy.spatial import distance
from gurobipy import Model, GRB
from itertools import combinations
import math
import time


def create_geometry():
    """Create and return the building geometries, obstacles, and buffer zones."""
    # Coordinates of the building
    polygon1 = Polygon([(0, 0), (10, 0), (10, 10), (25, 10), (25, 0), (40, 0), (40, 20), (35, 20),
        (35, 35), (40, 35), (40, 50), (30, 50), (30, 40), (15, 40), (15, 50), (0, 50), (0, 35),
        (10, 35), (10, 20), (0, 20), (0, 0)])

    polygon2 = Polygon([(60, 0), (70, 0), (70, 10), (85, 10), (85, 0), (100, 0), (100, 20), (95, 20),
        (95, 35), (100, 35), (100, 50), (90, 50), (90, 40), (75, 40), (75, 50), (60, 50),
        (60, 35), (65, 35), (65, 20), (60, 20), (60, 0)])

    obst1 = Polygon([(15, 3), (20, 3), (20, 6), (15, 6), (15, 3)])
    obst2 = Polygon([(48, 33), (52, 33), (52, 40), (48, 40), (48, 33)])

    building = MultiPolygon([polygon1, polygon2])
    closer_buffer = building.buffer(1)
    outer_buffer = building.buffer(30)

    # Obstacles that might block the visibility to building
    obst_ra = MultiPolygon([obst1, obst2])
    obst_vis = MultiPolygon([obst2])
    
    return building, closer_buffer, outer_buffer, obst_ra, obst_vis


def generate_grid_points(outer_buffer, building, closer_buffer, obst_ra, grid_spacing):
    """
    Generate valid grid points within the buffer region using spatial filtering.
    Uses a more efficient approach by first quickly filtering points based on bounding box,
    then applying more expensive geometric operations only on candidate points.
    """
    xmin, ymin, xmax, ymax = outer_buffer.bounds
    
    # First, generate all candidate points within the bounding box
    candidate_points = []
    for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
        for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
            candidate_points.append(Point(x, y))
    
    print(f"Generated {len(candidate_points)} candidate points based on bounding box")
    
    # Apply geometric filtering in batches for better performance
    grid_points = []
    
    # Create shapely geometries for quick contains tests
    outer_region = outer_buffer.difference(building)  # Area between building and outer buffer
    obstacle_region = obst_ra
    
    for point in candidate_points:
        # Quick check if point is within outer_buffer but not in building (most points eliminated here)
        if outer_region.contains(point) and not obstacle_region.contains(point):
            # More expensive check for minimum distance to the building
            min_distance = min(poly.exterior.distance(point) for poly in closer_buffer.geoms)
            if min_distance >= 0:
                grid_points.append(point)
    
    print(f"After filtering, {len(grid_points)} valid grid points remain")
    return grid_points


def create_graph(grid_points, closer_buffer, obst_ra, grid_spacing):
    """
    Create a directed graph from grid points using spatial indexing for efficiency.
    Uses KD-Tree to find nearby points, drastically reducing the number of 
    distance calculations and geometric operations.
    """
    from scipy.spatial import KDTree
    
    G = nx.DiGraph()
    
    # Add nodes
    for i, point in enumerate(grid_points):
        G.add_node(i, pos=(point.x, point.y))
    
    # Create KD-Tree for spatial indexing
    points_array = np.array([(p.x, p.y) for p in grid_points])
    kdtree = KDTree(points_array)
    
    # Maximum edge distance
    max_edge_distance = grid_spacing * np.sqrt(2)
    
    # List to store edges
    E = []
    
    # Query KD-Tree for neighbors and create edges
    for i, point in enumerate(grid_points):
        # Find all points within max_edge_distance
        indices = kdtree.query_ball_point([point.x, point.y], max_edge_distance)
        
        for j in indices:
            # Skip self-loops
            if i == j:
                continue
            
            p1, p2 = grid_points[i], grid_points[j]
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            
            # Extra verification of distance (should be redundant but ensures correctness)
            if dist <= max_edge_distance:
                edge_line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                
                # Ensure edge does not touch or cross into the polygon or obstacles
                if not closer_buffer.intersects(edge_line) and not obst_ra.intersects(edge_line):
                    G.add_edge(i, j, weight=dist)
                    G.add_edge(j, i, weight=dist)
                    E.append((i, j, dist))
                    E.append((j, i, dist))
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Spatial indexing limited geometric tests to {len(indices) * len(grid_points)} potential edges")
    
    return G, E


def calculate_angle(vec1, vec2):
    """Calculate the angle between two vectors."""
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    angle_rad = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle_rad)


def create_building_segments(building, segment_size):
    """Create segments along the building boundary."""
    boundry_lines = [LineString(poly.exterior.coords) for poly in building.geoms]
    
    segments = []
    for boundry_line in boundry_lines:
        for i in range(0, int(boundry_line.length), segment_size):
            seg_start = boundry_line.interpolate(i)
            seg_end = boundry_line.interpolate(min(i + segment_size, boundry_line.length))
            segments.append((seg_start, seg_end))
    
    return segments


def calculate_combined_visibility(G, E, segments, building, obst_vis, particle_spacing=1.0, epsilon=1e-6):
    """
    Unified function to calculate all visibility-related information in a single pass:
    - segment-to-edge visibility
    - edge-to-segment visibility
    - particle-level visibility
    - visibility ratio factors (VRF)
    
    This eliminates redundant calculations and improves performance.
    """
    # Initialize dictionaries
    E_vars = {(i, j): dist for i, j, dist in E}  # Placeholder for the actual Gurobi variables
    segment_visibility = {}          # Which edges can see each segment
    edge_visibility = {}             # Which segments each edge can see
    edge_particle_visibility = {}    # Which segments each particle on an edge can see
    segment_visibility_particles = {seg_idx: [] for seg_idx in range(len(segments))}  # Which edges (via particles) can see each segment
    VRF = {}                        # Visibility Ratio Factor for each edge
    
    # Initialize edge_visibility
    for edge in E_vars.keys():
        edge_visibility[edge] = []
        edge_particle_visibility[edge] = {}
    
    # Process each edge
    for edge in E_vars.keys():
        p1 = Point(G.nodes[edge[0]]['pos'])
        p2 = Point(G.nodes[edge[1]]['pos'])
        edge_line = LineString([p1, p2])
        edge_length = edge_line.length
        
        # Generate sample points along the edge (including endpoints)
        sample_points = [edge_line.interpolate(d) for d in np.arange(0, edge_length + 1e-6, particle_spacing)]
        if len(sample_points) < 2:  # Ensure we have at least the endpoints
            sample_points = [p1, p2]
        
        # Set to track all segments visible from any point on this edge
        all_visible_segments = set()
        
        # Process each particle (segment of the edge)
        for idx in range(len(sample_points) - 1):
            part_start = sample_points[idx]
            part_end = sample_points[idx+1]
            particle_vis = []  # Segments visible from this particle
            
            # Check visibility against each building segment
            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                # Calculate vectors for visibility determination
                vec1_start = (seg_start.x - part_start.x, seg_start.y - part_start.y)
                vec1_end = (seg_end.x - part_start.x, seg_end.y - part_start.y)
                vec2_start = (seg_start.x - part_end.x, seg_start.y - part_end.y)
                vec2_end = (seg_end.x - part_end.x, seg_end.y - part_end.y)
                segment_vec = (seg_end.x - seg_start.x, seg_end.y - seg_start.y)
                
                # Calculate angles
                angle1_start = calculate_angle(vec1_start, segment_vec)
                angle1_end = calculate_angle(vec1_end, segment_vec)
                angle2_start = calculate_angle(vec2_start, segment_vec)
                angle2_end = calculate_angle(vec2_end, segment_vec)
                
                # Calculate distances
                d1_start = part_start.distance(seg_start)
                d1_end = part_start.distance(seg_end)
                d2_start = part_end.distance(seg_start)
                d2_end = part_end.distance(seg_end)
                
                # Check line visibility
                line1_start = LineString([part_start, seg_start])
                line1_end = LineString([part_start, seg_end])
                line2_start = LineString([part_end, seg_start])
                line2_end = LineString([part_end, seg_end])
                
                # Check if lines touch the building
                touches1_start = line1_start.touches(building)
                touches1_end = line1_end.touches(building)
                touches2_start = line2_start.touches(building)
                touches2_end = line2_end.touches(building)
                
                # Skip if blocked by obstacles
                if ((line1_start.intersects(obst_vis) or line1_end.intersects(obst_vis)) and
                    (line2_start.intersects(obst_vis) or line2_end.intersects(obst_vis))):
                    continue
                
                # Check visibility conditions
                is_visible = ((6 <= d1_start <= 15 and 6 <= d1_end <= 15 and 
                             30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and 
                             touches1_start and touches1_end) or
                            (6 <= d2_start <= 15 and 6 <= d2_end <= 15 and 
                             30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and 
                             touches2_start and touches2_end))
                
                if is_visible:
                    # Update particle visibility
                    particle_vis.append(seg_idx)
                    
                    # Add to all visible segments for this edge
                    all_visible_segments.add(seg_idx)
                    
                    # If this is the first particle on the edge that sees this segment,
                    # update the segment-to-edge visibility mapping
                    if edge not in segment_visibility_particles[seg_idx]:
                        segment_visibility_particles[seg_idx].append(edge)
                    
                    # Special case for endpoints (p1 and p2) - update segment and edge visibility
                    if idx == 0 and part_start == p1:
                        if seg_idx not in edge_visibility[edge]:
                            edge_visibility[edge].append(seg_idx)
                            
                            # Initialize segment visibility if needed
                            if seg_idx not in segment_visibility:
                                segment_visibility[seg_idx] = []
                            
                            # Add this edge to segment visibility
                            if edge not in segment_visibility[seg_idx]:
                                segment_visibility[seg_idx].append(edge)
                    
                    if idx == len(sample_points) - 2 and part_end == p2:
                        if seg_idx not in edge_visibility[edge]:
                            edge_visibility[edge].append(seg_idx)
                            
                            # Initialize segment visibility if needed
                            if seg_idx not in segment_visibility:
                                segment_visibility[seg_idx] = []
                            
                            # Add this edge to segment visibility
                            if edge not in segment_visibility[seg_idx]:
                                segment_visibility[seg_idx].append(edge)
            
            # Store particle visibility results
            edge_particle_visibility[edge][idx] = particle_vis
        
        # Calculate VRF
        VRF[edge] = len(all_visible_segments) / (edge_length + epsilon)
    
    # Ensure all segments are in segment_visibility
    for seg_idx in range(len(segments)):
        if seg_idx not in segment_visibility:
            segment_visibility[seg_idx] = []
    
    return segment_visibility, edge_visibility, edge_particle_visibility, segment_visibility_particles, VRF


def create_optimization_model(G, E, E_vars, cost, segment_visibility_particles, VRF, tie_points, epsilon=1e-6):
    """Create and return a Gurobi optimization model."""
    model = Model("Model")
    
    # Create a cost dictionary with proper keys
    cost_dict = {}
    for i, j, dist in E:
        cost_dict[(i, j)] = dist
    
    # Edge variables
    E_vars = {}  # Clear any existing entries and create fresh variables
    for i, j, dist in E:
        E_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"edge_{i}_{j}")
    
    # Objective
    model.setObjective(
        sum(E_vars[edge] * cost_dict[edge] * (1/(VRF[edge] + epsilon)) for edge in E_vars), 
        GRB.MINIMIZE
    )
    
    # Constraint: All segments must be visible
    for seg_idx, edges in segment_visibility_particles.items():
        if edges:
            model.addConstr(sum(E_vars[edge] for edge in edges) >= 1, name=f"seg_visibility_{seg_idx}")
    
    # Constraint: incoming and outgoing edges on one node must be equal
    for node in G.nodes:
        in_degree = sum(E_vars[(i, node)] for i, _ in G.in_edges(node) if (i, node) in E_vars)
        out_degree = sum(E_vars[(node, j)] for _, j in G.out_edges(node) if (node, j) in E_vars)
        model.addConstr(in_degree == out_degree, name=f"flow_{node}")
    
    # Tie points constraints
    for node in tie_points:
        in_degree = sum(E_vars[(i, node)] for i, _ in G.in_edges(node) if (i, node) in E_vars)
        out_degree = sum(E_vars[(node, j)] for _, j in G.out_edges(node) if (node, j) in E_vars)
        model.addConstr(in_degree >= 2, name=f"tiepoint_in_{node}")
        model.addConstr(out_degree >= 2, name=f"tiepoint_out_{node}")
    
    # Pre-elimination for 3 edges subtours
    for U in combinations(G.nodes, 3):
        expr = 0
        for i in U:
            for j in U:
                if i != j and (i, j) in E_vars:
                    expr += E_vars[(i, j)]
        model.addConstr(expr <= 2, name=f"subtour3_{U}")
    
    # Enable lazy constraints
    model.Params.LazyConstraints = 1
    
    return model, E_vars


def get_subtour(nodes, selected_edges):
    """Identify subtours in the current solution."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(selected_edges)
    comps = list(nx.connected_components(H))
    
    # One component = no subtour
    if len(comps) == 1:
        return None
    return min(comps, key=len)


def subtourelim_callback(model, where):
    """Callback function to eliminate subtours during optimization."""
    if where == GRB.Callback.MIPSOL:
        sol = model.cbGetSolution(model._vars)
        selected = [(i, j) for (i, j) in model._vars if sol[i, j] > 0.5]
        
        used_nodes = set()
        for i, j in selected:
            used_nodes.add(i)
            used_nodes.add(j)
        used_nodes = list(used_nodes)
        
        subtour = get_subtour(used_nodes, selected)
        if subtour is not None:
            # Lazy constraint: the sum of edges inside the subtour must be <= |subtour| - 1
            expr = 0
            for i in subtour:
                for j in subtour:
                    if (i, j) in model._vars:
                        expr += model._vars[(i, j)]
            model.cbLazy(expr <= len(subtour) - 1)


def visualize_results(G, building, obst_ra, segments, selected_edges):
    """Visualize the building, obstacles, segments, and solution path."""
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=30, node_color='blue', edge_color='gray', font_size=8)
    
    # Building
    for poly in building.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y, 'g-', linewidth=2)
    
    # Obstacles
    for poly in obst_ra.geoms:
        x, y = poly.exterior.xy
        plt.fill(x, y, color='black', alpha=0.7)
    
    # Segment endpoints and segments id label
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        plt.plot(seg_start.x, seg_start.y, 'ro', markersize=3)
        plt.plot(seg_end.x, seg_end.y, 'ro', markersize=3)
        midpoint_x = (seg_start.x + seg_end.x) / 2.0
        midpoint_y = (seg_start.y + seg_end.y) / 2.0
        plt.text(midpoint_x, midpoint_y, str(seg_idx), fontsize=8, color='green')
    
    # The result path
    nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=2)
    
    plt.title("Selected Edges")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()


def main():
    """Main function to coordinate the path planning process."""
    start_time = time.time()
    
    # Create geometry
    building, closer_buffer, outer_buffer, obst_ra, obst_vis = create_geometry()
    
    # Generate grid points
    grid_points = generate_grid_points(outer_buffer, building, closer_buffer, obst_ra, grid_spacing=8)
    
    # Create graph
    G, E = create_graph(grid_points, closer_buffer, obst_ra, grid_spacing=8)
    
    # Create building segments
    segments = create_building_segments(building, segment_size=5)
    
    # Calculate all visibility information in a single pass (merged calculation)
    # Changed particle spacing from 1.0 to 4.0 to reduce visibility computation
    segment_visibility, edge_visibility, edge_particle_visibility, segment_visibility_particles, VRF = calculate_combined_visibility(
        G, E, segments, building, obst_vis, particle_spacing=4.0
    )
    
    mid_time = time.time()
    print(f"Pre-Processing Time: {mid_time - start_time:.2f} seconds")
    
    # Create optimization model - we pass an empty dictionary for E_vars since we'll create it in the function
    tie_points = [91]  # Node id of the selected tie points
    model, E_vars = create_optimization_model(G, E, {}, None, segment_visibility_particles, VRF, tie_points)
    
    # Set up model attributes for callbacks
    model._vars = E_vars
    model._nodes = list(G.nodes)
    
    # Optimize the model
    model.optimize(subtourelim_callback)
    
    # Extract solution
    selected_edges = [edge for edge in E_vars if E_vars[edge].X > 0.5]
    
    # Visualize results
    visualize_results(G, building, obst_ra, segments, selected_edges)
    
    end_time = time.time()
    print(f"Optimizing Time: {end_time - mid_time:.2f} seconds")
    print(f"Total Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()