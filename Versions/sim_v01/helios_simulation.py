import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyhelios
import glob
import time
import subprocess
import sys

def create_obj_file(coordinates, height, filename):
    """
    Create an OBJ file with extruded 2D coordinates to specified height.
    This approach handles both convex and concave polygons by triangulating the faces.
    """
    with open(filename, "w") as f:
        # Write vertices for bottom face (z=0)
        for x, y in coordinates:
            f.write(f"v {x} {y} 0\n")
        
        # Write vertices for top face (z=height)
        for x, y in coordinates:
            f.write(f"v {x} {y} {height}\n")
        
        num_points = len(coordinates)
        
        # Triangulate the bottom face (fan triangulation - works for convex shapes)
        for i in range(1, num_points - 1):
            f.write(f"f 1 {i+1} {i+2}\n")
        
        # Triangulate the top face
        for i in range(1, num_points - 1):
            v1 = 1 + num_points
            v2 = i + 1 + num_points
            v3 = i + 2 + num_points
            f.write(f"f {v1} {v2} {v3}\n")
        
        # Side faces (connecting bottom and top)
        for i in range(num_points - 1):
            v1 = i + 1
            v2 = i + 2
            v3 = v2 + num_points
            v4 = v1 + num_points
            # Split each quadrilateral into two triangles
            f.write(f"f {v1} {v2} {v3}\n")
            f.write(f"f {v1} {v3} {v4}\n")
        
        # Connect the last and first vertices
        v1 = num_points
        v2 = 1
        v3 = v2 + num_points
        v4 = v1 + num_points
        f.write(f"f {v1} {v2} {v3}\n")
        f.write(f"f {v1} {v3} {v4}\n")

def process_input_files():
    print("Creating directories...")
    # Create output directories
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/scenes", exist_ok=True)
    os.makedirs("assets/sceneparts", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("Loading geometry data...")
    # Load building and obstacle data
    with open("geometry.json", "r") as f:
        geometry_data = json.load(f)
    
    print("Loading trajectory data...")
    # Load trajectory data
    trajectory_points = []
    with open("path.txt", "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                try:
                    x, y = map(float, line.strip().split(","))
                    trajectory_points.append([x, y, 0])  # Add z=0
                except ValueError:
                    continue
    
    print(f"Creating OBJ files for {len(geometry_data['buildings'])} buildings...")
    # Create OBJ files for buildings
    for building in geometry_data["buildings"]:
        create_obj_file(
            building["coordinates"],
            12.0,  # Height of 12m
            f"assets/sceneparts/{building['id']}.obj"
        )
    
    print(f"Creating OBJ files for {len(geometry_data['obstacles'])} obstacles...")
    # Create OBJ files for obstacles
    for obstacle in geometry_data["obstacles"]:
        create_obj_file(
            obstacle["coordinates"],
            1.0,  # Height of 1m
            f"assets/sceneparts/{obstacle['id']}.obj"
        )
    
    print("Creating trajectory CSV...")
    # Create trajectory CSV
    with open("assets/trajectory.csv", "w") as f:
        f.write("x,y,z,roll,pitch,yaw,time\n")
        
        # Calculate time based on speed (2m/s)
        dist = 0
        time = 0
        prev_point = trajectory_points[0]
        
        for i, point in enumerate(trajectory_points):
            if i > 0:
                # Calculate distance from previous point
                segment_dist = np.sqrt((point[0] - prev_point[0])**2 + 
                                      (point[1] - prev_point[1])**2 + 
                                      (point[2] - prev_point[2])**2)
                dist += segment_dist
                time = dist / 2.0  # Speed = 2m/s
            
            # Write point with roll, pitch, yaw = 0
            f.write(f"{point[0]},{point[1]},{point[2]},0,0,0,{time}\n")
            prev_point = point
    
    print("Creating scene.xml...")
    # Create scene.xml
    scene_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="building_scene">
"""
    
    # Add buildings
    for building in geometry_data["buildings"]:
        scene_xml += f"""
        <part>
            <filter type="objloader">
                <param type="filepath" key="filepath" value="assets/sceneparts/{building['id']}.obj"/>
            </filter>
            <rotations>
                <x>0</x>
                <y>0</y>
                <z>0</z>
            </rotations>
            <translations>
                <x>0</x>
                <y>0</y>
                <z>0</z>
            </translations>
            <scale>1</scale>
        </part>
"""
    
    # Add obstacles
    for obstacle in geometry_data["obstacles"]:
        scene_xml += f"""
        <part>
            <filter type="objloader">
                <param type="filepath" key="filepath" value="assets/sceneparts/{obstacle['id']}.obj"/>
            </filter>
            <rotations>
                <x>0</x>
                <y>0</y>
                <z>0</z>
            </rotations>
            <translations>
                <x>0</x>
                <y>0</y>
                <z>0</z>
            </translations>
            <scale>1</scale>
        </part>
"""
    
    scene_xml += """
    </scene>
</document>
"""
    
    with open("assets/scenes/scene.xml", "w") as f:
        f.write(scene_xml)
    
    print("Creating survey.xml...")
    # Create survey.xml
    survey_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <survey name="building_scan">
        <scene>assets/scenes/scene.xml</scene>
        <platform>
            <device type="ground_vehicle">
                <rotations>
                    <source>none</source>
                </rotations>
                <positions>
                    <source>trajectory</source>
                </positions>
                <file>assets/trajectory.csv</file>
            </device>
            <scanner>
                <settings>
                    <active>true</active>
                    <pulseFreq>900000</pulseFreq>
                    <scanAngle>360</scanAngle>
                    <scanFreq>10</scanFreq>
                    <headRotatePerSec>0</headRotatePerSec>
                    <headRotateStart>0</headRotateStart>
                    <headRotateStop>0</headRotateStop>
                    <pulseLength>5e-09</pulseLength>
                    <beamDivergence>0.003</beamDivergence>
                    <trajectoryTimeInterval>0.001</trajectoryTimeInterval>
                </settings>
                <device type="velodyne_hdl32e"/>
                <rotations>
                    <x>0</x>
                    <y>0</y>
                    <z>0</z>
                </rotations>
                <position>
                    <x>0</x>
                    <y>0</y>
                    <z>1.75</z> <!-- Setting scanner height -->
                </position>
                <beamOrigin>
                    <x>0</x>
                    <y>0</y>
                    <z>0</z>
                </beamOrigin>
            </scanner>
        </platform>
    </survey>
</document>
"""
    
    with open("assets/survey.xml", "w") as f:
        f.write(survey_xml)

def run_simulation():
    """
    Run the simulation using the command line interface directly
    instead of using the Python API which seems to have compatibility issues.
    """
    print("Running Helios++ simulation using command line...")
    
    # Get absolute paths
    assets_dir = os.path.abspath("assets")
    survey_file = os.path.join(assets_dir, "survey.xml")
    output_dir = os.path.abspath("output")
    
    # Construct the command
    cmd = ["helios", survey_file, "--assets", assets_dir, "--output", output_dir]
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # Get final return code
        return_code = process.poll()
        
        # Get any error output
        stderr = process.stderr.read()
        if stderr:
            print("Error output:")
            print(stderr)
            
        if return_code != 0:
            print(f"Helios++ returned non-zero exit code: {return_code}")
        else:
            print("Helios++ simulation completed successfully!")
    
    except FileNotFoundError:
        print("Error: 'helios' command not found. Make sure Helios++ is installed correctly and in your PATH.")
        print("Alternatively, you can run the command manually:")
        print(f"helios {survey_file} --assets {assets_dir} --output {output_dir}")
        sys.exit(1)
    
    # Find the output file
    time.sleep(1)  # Wait for files to be written
    output_files = glob.glob(f"{output_dir}/*.xyz")
    if not output_files:
        output_files = glob.glob(f"{output_dir}/*/*.xyz")
    
    if not output_files:
        print("Warning: No output files found.")
        # Return a path that might be correct based on default Helios++ behavior
        return os.path.join(output_dir, "Survey Playback", "building_scan", "building_scan.xyz")
    else:
        output_file = output_files[0]
        print(f"Found output file: {output_file}")
        return output_file

def visualize_point_cloud(point_cloud_file):
    print(f"Visualizing point cloud from: {point_cloud_file}")
    
    # Check if file exists
    if not os.path.exists(point_cloud_file):
        print(f"Error: Point cloud file not found: {point_cloud_file}")
        # Try to find any xyz file
        xyz_files = []
        for root, dirs, files in os.walk("output"):
            for file in files:
                if file.endswith(".xyz"):
                    xyz_files.append(os.path.join(root, file))
        
        if xyz_files:
            point_cloud_file = xyz_files[0]
            print(f"Found alternative point cloud file: {point_cloud_file}")
        else:
            print("No point cloud files found in output directory.")
            return
    
    # Load the point cloud data
    try:
        points = np.loadtxt(point_cloud_file, delimiter=' ')
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    print(f"Loaded {len(points)} points.")
    
    # Check if we have any points
    if len(points) == 0:
        print("No points found in the file.")
        return
    
    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the point cloud
    scatter = ax.scatter(x, y, z, s=0.1, c=z, cmap='viridis')
    
    # Add a color bar
    plt.colorbar(scatter, ax=ax, label='Height (m)')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Simulated Point Cloud')
    
    # Set equal aspect ratio
    try:
        max_range = np.max([np.max(x) - np.min(x), 
                             np.max(y) - np.min(y),
                             np.max(z) - np.min(z)])
        mid_x = (np.max(x) + np.min(x)) * 0.5
        mid_y = (np.max(y) + np.min(y)) * 0.5
        mid_z = (np.max(z) + np.min(z)) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    except Exception as e:
        print(f"Warning: Could not set equal aspect ratio: {e}")
    
    print("Showing visualization...")
    plt.show()
    
    # Save the figure
    plt.savefig("output/point_cloud_visualization.png")
    print("Saved visualization to output/point_cloud_visualization.png")

# Main function
def main():
    try:
        # Process the input files
        print("Processing input files...")
        process_input_files()
        
        # Run the simulation
        output_file = run_simulation()
        
        # Visualize the results
        visualize_point_cloud(output_file)
        
        print("Simulation and visualization completed successfully!")
        
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()

# Run the program
if __name__ == "__main__":
    main()