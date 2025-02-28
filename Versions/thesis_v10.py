import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from scipy.spatial import distance
from gurobipy import Model, GRB
from itertools import combinations
import math

# Coordinates of the building
polygon1 = Polygon ([(0, 0), (10, 0), (10, 10), (25,10), (25, 0), (40, 0), (40, 20), (35, 20),
    (35, 35), (40, 35), (40, 50), (30, 50), (30, 40), (15, 40), (15, 50), (0, 50), (0, 35),
    (10, 35), (10, 20), (0, 20), (0, 0) ])

polygon2 = Polygon ([(60, 0), (70, 0), (70, 10), (85, 10), (85, 0), (100, 0), (100, 20), (95, 20),
    (95, 35), (100, 35), (100, 50), (90, 50), (90, 40), (75, 40), (75, 50), (60, 50),
    (60, 35), (65, 35), (65, 20), (60, 20), (60, 0)])

obst1 = Polygon ([(15, -8), (20, -8), (20, 0), (15,0), (15, -8)])
obst2 = Polygon ([(48, 33), (52, 33), (52, 40), (48, 40), (48, 33)])

building= MultiPolygon([polygon1, polygon2])
closer_buffer =building.buffer(1)  
outer_buffer =building.buffer(30)  

# Obstacles that might blovk the visibility to building
obstacles= MultiPolygon([obst1, obst2])

xmin, ymin, xmax, ymax = closer_buffer.bounds


# Grod points
grid_spacing = 8

xmin, ymin, xmax, ymax = outer_buffer.bounds

grid_points = []
for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
    for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
        point = Point(x, y)
        
        if (not building.contains(point) and outer_buffer.contains(point)
                and not obstacles.contains(point)):  
            min_distance = min(poly.exterior.distance(point) for poly in closer_buffer.geoms)
            if min_distance >= 0:
                grid_points.append(point)


# The Graph
G = nx.DiGraph()

# Add nodes
for i, point in enumerate(grid_points):
    G.add_node(i, pos=(point.x, point.y))


max_edge_distance = grid_spacing * np.sqrt(2)
E = []
for i, p1 in enumerate(grid_points):
    for j, p2 in enumerate(grid_points):
        if i != j:
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            if dist <= max_edge_distance:
                
                edge_line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                
                # Ensure edge does not touch or cross into the polygon
                if not closer_buffer.intersects(edge_line) and not obstacles.intersects(edge_line):
                    G.add_edge(i, j, weight=dist)
                    G.add_edge(j, i, weight=dist)
                    E.append((i, j, dist))
                    E.append((j, i, dist))

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
boundry_lines = [LineString(poly.exterior.coords) for poly in building.geoms]

segment_size = 5
segments = []
for boundry_line in boundry_lines:
    for i in range(0, int(boundry_line.length), segment_size):
        seg_start = boundry_line.interpolate(i)
        seg_end = boundry_line.interpolate(min(i + segment_size, boundry_line.length))
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

        touches1_start = line1_start.touches(building)
        touches1_end = line1_end.touches(building)
        touches2_start = line2_start.touches(building)
        touches2_end = line2_end.touches(building)
        
        if ((line1_start.intersects(obstacles) or line1_end.intersects(obstacles)) and
            (line2_start.intersects(obstacles) or line2_end.intersects(obstacles))):
            continue  # Skip this edge for the current segment if any line touches an obstacle


        if seg_idx not in segment_visibility:
            segment_visibility[seg_idx] = []

        if ((6 <= d1_start <= 15 and 6 <= d1_end <= 15 and 30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and touches1_start and touches1_end) or
            (6 <= d2_start <= 15 and 6 <= d2_end <= 15 and 30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and touches2_start and touches2_end)):
            segment_visibility[seg_idx].append(edge)
            edge_visibility[edge].append(seg_idx)
   
            
# %% Objectives and Constraints

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


# Node id of the selected tie points
# Path will pass these points more equal or more than 2 times
tie_points = [99]  

for node in tie_points:
    in_degree = sum(E_vars[(i, node)] for i, _ in G.in_edges(node) if (i, node) in E_vars)
    out_degree = sum(E_vars[(node, j)] for _, j in G.out_edges(node) if (node, j) in E_vars)
    model.addConstr(in_degree >= 2, name=f"tiepoint_in_{node}")
    model.addConstr(out_degree >= 2, name=f"tiepoint_out_{node}")


# pre-elimination for 3 edges subtours
for U in combinations(G.nodes, 3):
    expr = 0
    for i in U:
        for j in U:
            if i != j and (i, j) in E_vars:
                expr += E_vars[(i, j)]
    model.addConstr(expr <= 2, name=f"subtour3_{U}")

#Enabling lazy constraints:
model.Params.LazyConstraints = 1

model._vars = E_vars
model._nodes = list(G.nodes)

def get_subtour(nodes, selected_edges):
    
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(selected_edges)
    comps = list(nx.connected_components(H))
    
    # one compom=nent = no subtour
    if len(comps) == 1:
        return None
    return min(comps, key=len)

def subtourelim_callback(model, where):
   
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
            # lazy constraint: the sum of edges inside the subtour must be <= |subtour| - 1.
            expr = 0
            for i in subtour:
                for j in subtour:
                    if (i, j) in model._vars:
                        expr += model._vars[(i, j)]
            model.cbLazy(expr <= len(subtour) - 1)


model.optimize(subtourelim_callback)

# %% Plot

# Graph nodes and edges
plt.figure(figsize=(10, 10))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=30, node_color='blue', edge_color='gray', font_size=8)

# Building
for poly in building.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, 'g-', linewidth=2)

# Obstacles    
for poly in obstacles.geoms:
    x, y = poly.exterior.xy
    plt.fill(x, y, color='black', alpha=0.7)
    
# Segment endpoints and segemnts id label
for seg_idx, (seg_start, seg_end) in enumerate(segments):
    plt.plot(seg_start.x, seg_start.y, 'ro', markersize=3)
    plt.plot(seg_end.x, seg_end.y, 'ro', markersize=3)
    midpoint_x = (seg_start.x + seg_end.x) / 2.0
    midpoint_y = (seg_start.y + seg_end.y) / 2.0
    plt.text(midpoint_x, midpoint_y, str(seg_idx), fontsize=8, color='green')


# The result path
selected_edges = [edge for edge in E_vars if E_vars[edge].X > 0.5]
nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=2)

plt.title("Selected Edges")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()

