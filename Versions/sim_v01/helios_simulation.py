import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import time
import subprocess
import sys

def create_trajectory_csv():
    """Create the trajectory CSV from path.txt file"""
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
    
    return

def create_config_files():
    """Create scene.xml and survey.xml using existing OBJ files"""
    print("Creating XML configuration files...")
    
    # Create output directories
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/scenes", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Find existing OBJ files
    obj_files = []
    for filename in os.listdir("assets/sceneparts"):
        if filename.endswith(".obj"):
            obj_name = os.path.splitext(filename)[0]
            obj_files.append(obj_name)
    
    print(f"Found {len(obj_files)} OBJ files: {', '.join(obj_files)}")
    
    # Create scene.xml
    print("Creating scene.xml...")
    scene_xml = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="building_scene">
"""
    
    # Add all existing OBJ files to the scene
    for obj_name in obj_files:
        scene_xml += f"""
        <part>
            <filter type="objloader">
                <param type="filepath" key="filepath" value="assets/sceneparts/{obj_name}.obj"/>
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
    
    # Create a simplified survey.xml that uses default scanners
    print("Creating simplified survey.xml...")
    survey_xml = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <survey name="building_scan">
        <scene>assets/scenes/scene.xml</scene>
        <leg>
            <platform>
                <type>ground_vehicle</type>
                <track_file>assets/trajectory.csv</track_file>
                <scanner>
                    <type>velodyne_vlp16</type>
                    <active>true</active>
                    <pulseFreq_hz>300000</pulseFreq_hz>
                    <scanAngle_deg>360</scanAngle_deg>
                    <scanFreq_hz>10</scanFreq_hz>
                    <beamDivergence_rad>0.003</beamDivergence_rad>
                    <position>
                        <x>0</x>
                        <y>0</y>
                        <z>1.75</z>
                    </position>
                </scanner>
            </platform>
        </leg>
    </survey>
</document>
"""
    
    with open("assets/survey.xml", "w") as f:
        f.write(survey_xml)

def run_simulation():
    """
    Run the simulation using the command line interface directly
    """
    print("Running Helios++ simulation using command line...")
    
    # Get absolute paths
    assets_dir = os.path.abspath("assets")
    survey_file = os.path.join(assets_dir, "survey.xml")
    output_dir = os.path.abspath("output")
    
    # Get Helios++ installation directory to include in assets path
    try:
        # Try to find where Helios++ is installed (default data location)
        import site
        site_packages = site.getsitepackages()
        helios_paths = []
        for site_path in site_packages:
            if os.path.exists(os.path.join(site_path, 'pyhelios', 'data')):
                helios_paths.append(os.path.join(site_path, 'pyhelios', 'data'))
        
        # Add Python package data directory to assets paths
        assets_paths = [assets_dir]
        assets_paths.extend(helios_paths)
        
        print(f"Using these asset paths: {assets_paths}")
    except:
        # Fallback if we can't identify the site packages
        assets_paths = [assets_dir]
    
    # Construct the command with multiple assets paths
    cmd = ["helios", survey_file, "--output", output_dir]
    
    # Add asset paths
    for path in assets_paths:
        cmd.extend(["--assets", path])
    
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
        # Try to find any output file recursively
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".xyz"):
                    output_files.append(os.path.join(root, file))
                    break
                    
        if not output_files:
            print("Warning: No output files found.")
            # Return a path that might be correct based on default Helios++ behavior
            return os.path.join(output_dir, "Survey Playback", "building_scan", "building_scan.xyz")
    
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
    
    # Plot the point cloud with small point size
    scatter = ax.scatter(x, y, z, s=0.05, c=z, cmap='viridis')
    
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
    
    # Save the figure
    plt.savefig("output/point_cloud_visualization.png")
    print("Saved visualization to output/point_cloud_visualization.png")
    
    # Show the plot
    print("Showing visualization...")
    plt.show()

# Main function
def main():
    try:
        # Create trajectory CSV
        create_trajectory_csv()
        
        # Create XML config files
        create_config_files()
        
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