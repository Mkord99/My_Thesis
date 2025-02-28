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

point_index_map = {}
for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
    for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
        point = Point(x, y)
        if not polygon.contains(point) and not polygon.touches(point):
            point_index_map[(x, y)] = len(grid_points)
            grid_points.append(point)

# Building boundary segmentation
boundary_line = LineString(polygon.exterior.coords)
segment_size = 5
segments = []
for i in range(0, int(boundary_line.length), segment_size):
    segment = boundary_line.interpolate(i), boundary_line.interpolate(min(i + segment_size, boundary_line.length))
    segments.append(segment)

# Create edges between grid points
edges = []

max_edge_distance = grid_spacing * np.sqrt(2)
for i, p1 in enumerate(grid_points):
    for j, p2 in enumerate(grid_points):
        if i != j:
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            if dist <= max_edge_distance:
                edges.append((i, j, dist))

# Function to calculate the angle
def calculate_angle(vec1, vec2):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    angle_rad = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle_rad)

# Identify edges meeting visibility constraints
visible_edges = set()
for seg_idx, (seg_start, seg_end) in enumerate(segments):
    for i, j, dist in edges:
        p1, p2 = grid_points[i], grid_points[j]

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
        if ((6 <= d1_start <= 12 and 6 <= d1_end <= 12 and 30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and touches1_start and touches1_end) or
            (6 <= d2_start <= 12 and 6 <= d2_end <= 12 and 30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and touches2_start and touches2_end)):
            visible_edges.add((i, j))

weights = {}  
for i, j, dist in edges:
    if (i, j) in visible_edges:
        weights[(i, j)] = dist  

start_point = Point(-14, -16)
end_point = Point(38, 40)

if (start_point.x, start_point.y) not in point_index_map or (end_point.x, end_point.y) not in point_index_map:
    raise ValueError("Start or end point are not in the grid set")

start_index = point_index_map[(start_point.x, start_point.y)]
end_index = point_index_map[(end_point.x, end_point.y)]


model = Model("optimized_path")

flow_vars = {}
for i, j in visible_edges:
    flow_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"flow_{i}_{j}")


for k in range(len(grid_points)):
    inflow = quicksum(flow_vars[(i, j)] for i, j in flow_vars if j == k)
    outflow = quicksum(flow_vars[(i, j)] for i, j in flow_vars if i == k)
    if k == start_index:
        model.addConstr(outflow - inflow == 1, name=f"flow_start_{k}")
    elif k == end_index:
        model.addConstr(outflow - inflow == -1, name=f"flow_end_{k}")
    else:
        model.addConstr(outflow - inflow == 0, name=f"flow_conservation_{k}")


model.setObjective(quicksum(weights[edge] * flow_vars[edge] for edge in flow_vars), GRB.MINIMIZE)


model.optimize()


# %% Result Plot

x, y = polygon.exterior.xy
plt.plot(x, y, label='Polygon Boundary')


plt.scatter([p.x for p in grid_points], [p.y for p in grid_points], s=10, label='Grid Points')


'''

for i, j, _ in edges:
    p1, p2 = grid_points[i], grid_points[j]
    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'blue', lw=1, alpha=0.5)
'''

for i, j in visible_edges:
    p1, p2 = grid_points[i], grid_points[j]
    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g', lw=1.5, alpha=0.8, label="Visible Edge")


for seg_start, seg_end in segments:
    plt.scatter([seg_start.x, seg_end.x], [seg_start.y, seg_end.y], color='orange', s=20, label="Segment Points")



if model.status == GRB.OPTIMAL:
    for i, j in flow_vars:
        if flow_vars[(i, j)].x > 0.5:
            p1, p2 = grid_points[i], grid_points[j]
            plt.plot([p1.x, p2.x], [p1.y, p2.y], 'r', lw=2, label="Selected Path")


plt.scatter([start_point.x], [start_point.y], color='blue', label="Start Point", zorder=5)
plt.scatter([end_point.x], [end_point.y], color='blue', label="End Point", zorder=5)


plt.title("Shortest Path with Visibility Constraints")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
#plt.legend()
plt.show()
