import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Load the JSON data from a file
with open('geometry.json', 'r') as file:
    data = json.load(file)

fig, ax = plt.subplots(figsize=(10, 10))

# Plot buildings
for building in data["buildings"]:
    coords = building["coordinates"]
    polygon = Polygon(coords, closed=True, edgecolor='black', facecolor='lightgray', linewidth=1.5, label='Building')
    ax.add_patch(polygon)
    # Optional: label each building
    centroid_x = sum([x for x, y in coords]) / len(coords)
    centroid_y = sum([y for x, y in coords]) / len(coords)
    ax.text(centroid_x, centroid_y, building["id"], ha='center', va='center', fontsize=9, weight='bold')

# Plot obstacles
for obs in data["obstacles"]:
    coords = obs["coordinates"]
    types = obs["type"]
    
    # Determine color based on type
    if "radiation" in types and "visibility" in types:
        color = 'purple'
    elif "radiation" in types:
        color = 'red'
    elif "visibility" in types:
        color = 'blue'
    else:
        color = 'gray'
    
    polygon = Polygon(coords, closed=True, edgecolor=color, facecolor=color, alpha=0.6, linewidth=1.5)
    ax.add_patch(polygon)
    # Optional: label each obstacle
    centroid_x = sum([x for x, y in coords]) / len(coords)
    centroid_y = sum([y for x, y in coords]) / len(coords)
    ax.text(centroid_x, centroid_y, obs["id"], ha='center', va='center', fontsize=8)

# Set plot limits and aspect ratio
ax.set_xlim(-10, 200)
ax.set_ylim(-10, 180)
ax.set_aspect('equal')
ax.set_title("Buildings and Obstacles Map")
plt.grid(True)
plt.show()
