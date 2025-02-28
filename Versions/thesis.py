import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import scale
import matplotlib.pyplot as plt
from scipy.spatial import distance
from gurobipy import Model, GRB, quicksum
import math

# Defining building coordinates
coordinates = [
    (0, 0),  
    (0, 20), 
    (20, 20),  
    (20, 15),  
    (10, 15),  
    (10, 5),  
    (20, 5),  
    (20, 0),  
    (0, 0)  
]

# Scaling the polygon
polygon = Polygon(coordinates)
polygon = scale(polygon, xfact=2, yfact=2, origin='centroid')
grid_spacing = 4
max_distance = 10


xmin, ymin, xmax, ymax = polygon.bounds
xmin -= max_distance
ymin -= max_distance
xmax += max_distance
ymax += max_distance

# Create grid points around the building
grid_points = []
for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
    for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
        point = Point(x, y)
        if not polygon.contains(point) and not polygon.touches(point):
            grid_points.append(point)

# Ploting the polygon and grid points
# x, y = polygon.exterior.xy
# plt.plot(x, y)
plt.scatter([p.x for p in grid_points], [p.y for p in grid_points], s=10)
plt.title("Polygon with Gridded Surface")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")

# Building boundary segmentation
boundary_line = LineString(polygon.exterior.coords)
segment_size = 5
segments = []
for i in range(0, int(boundary_line.length), segment_size):
    segment = boundary_line.interpolate(i), boundary_line.interpolate(min(i + segment_size, boundary_line.length))
    segments.append(segment)

# Plot segments on the building
for seg_start, seg_end in segments:
    plt.plot([seg_start.x, seg_end.x], [seg_start.y, seg_end.y], 'b', lw=2) 
    plt.plot(seg_start.x, seg_start.y, 'ro')
    plt.plot(seg_end.x, seg_end.y, 'ro') 

# Create edges between grid points
edges = []
max_edge_distance = grid_spacing * np.sqrt(2)
for i, p1 in enumerate(grid_points):
    for j, p2 in enumerate(grid_points):
        if i != j:
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            if dist <= max_edge_distance:
                edges.append((i, j, dist))

# Plot edges
for edge in edges:
    p1 = grid_points[edge[0]]
    p2 = grid_points[edge[1]]
    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'gray', alpha=0.5)

# Function to calculate the angle
def calculate_angle(vec1, vec2):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    angle_rad = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle_rad)

# Gurobi model
model = Model("visibility_optimization")

# Variables and weights
edge_vars = {}
weights = {}


for i, j, dist in edges:
    edge_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"edge_{i}_{j}")
    weights[(i, j)] = dist  # or assign based on your criteria


segment_visibility = {seg: [] for seg in range(len(segments))}
edge_visibility = {edge: [] for edge in edge_vars}


angles = {}

# Viability constraints
for seg_idx, (seg_start, seg_end) in enumerate(segments):
    for edge, var in edge_vars.items():
        p1, p2 = grid_points[edge[0]], grid_points[edge[1]]

        # Vectors from edge vertex to segment start and end point
        vec1_start = (seg_start.x - p1.x, seg_start.y - p1.y)
        vec1_end = (seg_end.x - p1.x, seg_end.y - p1.y)
        vec2_start = (seg_start.x - p2.x, seg_start.y - p2.y)
        vec2_end = (seg_end.x - p2.x, seg_end.y - p2.y)

        segment_vec = (seg_end.x - seg_start.x, seg_end.y - seg_start.y)

        # Angle calculations
        angle1_start = calculate_angle(vec1_start, segment_vec)
        angle1_end = calculate_angle(vec1_end, segment_vec)
        angle2_start = calculate_angle(vec2_start, segment_vec)
        angle2_end = calculate_angle(vec2_end, segment_vec)

        # Store calculated angles in the angles dictionary
        angles[(seg_idx, edge)] = {
            'angle1_start': angle1_start,
            'angle1_end': angle1_end,
            'angle2_start': angle2_start,
            'angle2_end': angle2_end
        }

        d1_start, d1_end = p1.distance(seg_start), p1.distance(seg_end)
        d2_start, d2_end = p2.distance(seg_start), p2.distance(seg_end)

        line1_start = LineString([p1, seg_start])
        line1_end = LineString([p1, seg_end])
        line2_start = LineString([p2, seg_start])
        line2_end = LineString([p2, seg_end])

        touches1_start = line1_start.touches(polygon)
        touches1_end = line1_end.touches(polygon)
        touches2_start = line2_start.touches(polygon)
        touches2_end = line2_end.touches(polygon)

        # The edge is valid if it touches at exactly one point for both vectors
        if ((6 <= d1_start <= 15 and 6 <= d1_end <= 15 and 30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and touches1_start and touches1_end) or
            (6 <= d2_start <= 15 and 6 <= d2_end <= 15 and 30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and touches2_start and touches2_end)):
            segment_visibility[seg_idx].append(edge)
            edge_visibility[edge].append(seg_idx)

# Constraint for each segment to be covered by at least one edge
for seg_idx, visible_edges in segment_visibility.items():
    model.addConstr(quicksum(edge_vars[edge] for edge in visible_edges) >= 1,
                    name=f"visibility_constraint_segment_{seg_idx}")

# Objective
model.setObjective(quicksum(edge_vars[edge] * weights[edge] for edge in edge_vars), GRB.MINIMIZE)


#model.optimize()


if model.status == GRB.OPTIMAL:
    selected_edges = [edge for edge, var in edge_vars.items() if var.x > 0.5]
    print("Selected edges:")
    for edge in selected_edges:
        print(edge)

    
    plt.plot(x, y)
    plt.scatter([p.x for p in grid_points], [p.y for p in grid_points], s=10)
    for edge in selected_edges:
        p1 = grid_points[edge[0]]
        p2 = grid_points[edge[1]]
        plt.plot([p1.x, p2.x], [p1.y, p2.y], 'r', lw=2)
    plt.show()

else:
    print("Optimization failed")
    
    
    
    
    
    
    
    
    
