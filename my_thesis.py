import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import scale
from scipy.spatial import distance
from gurobipy import Model, GRB
import math

# Coordinates of the building
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

# Grod points
grid_points = []
for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
    for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
        point = Point(x, y)
        if not polygon.contains(point) and not polygon.touches(point):
            grid_points.append(point)

# The Graph
G = nx.DiGraph()

# Nodes
for i, point in enumerate(grid_points):
    G.add_node(i, pos=(point.x, point.y))

# Directed edges between nodes 
max_edge_distance = grid_spacing * np.sqrt(2)
E = []
for i, p1 in enumerate(grid_points):
    for j, p2 in enumerate(grid_points):
        if i != j:
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            if dist <= max_edge_distance:
                G.add_edge(i, j, weight=dist)
                G.add_edge(j, i, weight=dist)  
                E.append((i, j, dist))
                E.append((j, i, dist))  

# Gurobi model
model = Model("Model")

# Edge variables and cost
E_vars = {}
cost = {}

for i, j, dist in E:
    E_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"edge_{i}_{j}")
    cost[(i, j)] = dist

# Segment visibility and edge visibility
segment_visibility = {}
edge_visibility = {edge: [] for edge in E_vars}
angles = {}

# Building boundary segmentation
boundary_line = LineString(polygon.exterior.coords)
segment_size = 5
segments = []
for i in range(0, int(boundary_line.length), segment_size):
    seg_start = boundary_line.interpolate(i)
    seg_end = boundary_line.interpolate(min(i + segment_size, boundary_line.length))
    segments.append((seg_start, seg_end))

# Function to calculate the angle
def calculate_angle(vec1, vec2):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    angle_rad = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle_rad)

# Visibility constraints
for seg_idx, (seg_start, seg_end) in enumerate(segments):
    for edge, var in E_vars.items():
        p1, p2 = Point(G.nodes[edge[0]]['pos']), Point(G.nodes[edge[1]]['pos'])
        
        
        vec1_start = (seg_start.x - p1.x, seg_start.y - p1.y)
        vec1_end = (seg_end.x - p1.x, seg_end.y - p1.y)
        vec2_start = (seg_start.x - p2.x, seg_start.y - p2.y)
        vec2_end = (seg_end.x - p2.x, seg_end.y - p2.y)

        segment_vec = (seg_end.x - seg_start.x, seg_end.y - seg_start.y)

        
        angle1_start = calculate_angle(vec1_start, segment_vec)
        angle1_end = calculate_angle(vec1_end, segment_vec)
        angle2_start = calculate_angle(vec2_start, segment_vec)
        angle2_end = calculate_angle(vec2_end, segment_vec)

        
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

        if seg_idx not in segment_visibility:
            segment_visibility[seg_idx] = []

        if ((6 <= d1_start <= 15 and 6 <= d1_end <= 15 and 30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and touches1_start and touches1_end) or
            (6 <= d2_start <= 15 and 6 <= d2_end <= 15 and 30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and touches2_start and touches2_end)):
            segment_visibility[seg_idx].append(edge)
            edge_visibility[edge].append(seg_idx)

# Objective
model.setObjective(sum(E_vars[(i, j)] * cost[(i, j)] for i, j, _ in E), GRB.MINIMIZE)

# Constraint: All segments must be visible
for seg_idx, edges in segment_visibility.items():
    if edges:
        model.addConstr(sum(E_vars[edge] for edge in edges) >= 1, name=f"seg_visibility_{seg_idx}")

# Constraint: incoming and outgoing edges on one node must be equal
for node in G.nodes:
    in_degree = sum(E_vars[(i, node)] for i, _ in G.in_edges(node) if (i, node) in E_vars)
    out_degree = sum(E_vars[(node, j)] for _, j in G.out_edges(node) if (node, j) in E_vars)

    model.addConstr(in_degree == out_degree, name=f"flow_{node}")

# Constraint: Only one direction between nodes can be selected
for i, j, _ in E:
    if (j, i) in E_vars:  
        model.addConstr(E_vars[(i, j)] + E_vars[(j, i)] <= 1, name=f"no_bidirectional_{i}_{j}")

model.optimize()
# %% Plot

plt.figure(figsize=(10, 10))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue', edge_color='gray', font_size=8)


x, y = polygon.exterior.xy
plt.plot(x, y, 'r-', linewidth=2)


selected_edges = [edge for edge in E_vars if E_vars[edge].X > 0.5]
nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=2)

plt.title("Selected Edges")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()
