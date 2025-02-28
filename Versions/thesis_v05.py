import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import scale
import matplotlib.pyplot as plt
from scipy.spatial import distance
from gurobipy import Model, GRB, quicksum
import math
import collections

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
V = []
for x in range(int(xmin), int(xmax) + grid_spacing, grid_spacing):
    for y in range(int(ymin), int(ymax) + grid_spacing, grid_spacing):
        point = Point(x, y)
        if not polygon.contains(point) and not polygon.touches(point):
            V.append(point)

# Plotting the polygon and grid points
plt.scatter([p.x for p in V], [p.y for p in V], s=10)
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

# Create E between grid points
E = []
max_edge_distance = grid_spacing * np.sqrt(2)
for i, p1 in enumerate(V):
    for j, p2 in enumerate(V):
        if i != j:
            dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
            if dist <= max_edge_distance:
                E.append((i, j, dist))

# Cost dictionary for each edge
cost = {}
for i, j, dist in E:
    cost[(i, j)] = dist

# Visibility calculations
segment_visibility = collections.defaultdict(list)  
edge_visibility = collections.defaultdict(list)  
angles = {}  

# Function to calculate the angle
def calculate_angle(vec1, vec2):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    angle_rad = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle_rad)

for seg_idx, (seg_start, seg_end) in enumerate(segments):
    for edge_idx, (i, j, _) in enumerate(E):
        p1, p2 = V[i], V[j]

        # Vectors from edge vertices to segment start and end points
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

        # Distance calculations
        d1_start, d1_end = p1.distance(seg_start), p1.distance(seg_end)
        d2_start, d2_end = p2.distance(seg_start), p2.distance(seg_end)

        # Lines connecting vertices to segment points
        line1_start = LineString([p1, seg_start])
        line1_end = LineString([p1, seg_end])
        line2_start = LineString([p2, seg_start])
        line2_end = LineString([p2, seg_end])

        # Check if lines touch the polygon boundary
        touches1_start = line1_start.touches(polygon)
        touches1_end = line1_end.touches(polygon)
        touches2_start = line2_start.touches(polygon)
        touches2_end = line2_end.touches(polygon)

        # Check if edge visibility conditions are satisfied
        if ((6 <= d1_start <= 15 and 6 <= d1_end <= 15 and 30 <= angle1_start <= 150 and 30 <= angle1_end <= 150 and touches1_start and touches1_end) or
            (6 <= d2_start <= 15 and 6 <= d2_end <= 15 and 30 <= angle2_start <= 150 and 30 <= angle2_end <= 150 and touches2_start and touches2_end)):
            # Add visibility relationship to dictionaries
            segment_visibility[seg_idx].append(edge_idx)
            edge_visibility[edge_idx].append(seg_idx)


model = Model("TSP with Visibility")

# Decision variables
x = model.addVars(cost.keys(), vtype=GRB.BINARY, name="x")

# Segment visibility variables: 1 if segment is visible, 0 otherwise
s = model.addVars(len(segments), vtype=GRB.BINARY, name="s")

# Objective function: Minimize total travel cost
model.setObjective(quicksum(cost[i, j] * x[i, j] for i, j in cost.keys()), GRB.MINIMIZE)

# Degree constraints: Each node must have exactly two edges
for i in range(len(V)):
    model.addConstr(quicksum(x[i, j] for j in range(len(V)) if (i, j) in cost) +
                    quicksum(x[j, i] for j in range(len(V)) if (j, i) in cost) == 2, name=f"degree_{i}")

# Symmetry constraint: Ensure x(i, j) - x(j, i) = 0
for i, j in cost.keys():
    model.addConstr(x[i, j] - x[j, i] == 0, name=f"symmetry_{i}_{j}")

# Visibility constraints: Ensure segments are visible
for seg_idx in range(len(segments)):
    model.addConstr(quicksum(x[i, j] for edge_idx in segment_visibility[seg_idx] for i, j, _ in [E[edge_idx]]) >= s[seg_idx], name=f"visibility_{seg_idx}")

# All segments must be visible
model.addConstr(quicksum(s[seg_idx] for seg_idx in range(len(segments))) == len(segments), name="all_segments_visible")

# Optimize the model
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    selected_edges = [(i, j) for i, j in cost.keys() if x[i, j].x > 0.5]
    print("Selected edges:", selected_edges)
    visible_segments = [seg_idx for seg_idx in range(len(segments)) if s[seg_idx].x > 0.5]
    print("Visible segments:", visible_segments)

    # Plot the selected edges
    for i, j in selected_edges:
        p1, p2 = V[i], V[j]
        plt.plot([p1.x, p2.x], [p1.y, p2.y], 'r-', lw=2)
else:
    print("No optimal solution found.")

# Show plot
plt.show()
