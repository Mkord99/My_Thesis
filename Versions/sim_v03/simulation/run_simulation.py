#!/usr/bin/env python3
"""
Helios++ Mobile Laser Scanning Simulation
Human with Heron Lite backpack scanner
"""

import os
import sys
import time
from datetime import datetime

# Import Helios++ Python bindings
try:
    import pyhelios
except ImportError:
    print("Error: pyhelios module not found!")
    print("Make sure Helios++ Python bindings are installed and in PYTHONPATH")
    sys.exit(1)

def run_simulation():
    """Run the mobile laser scanning simulation"""
    
    print("=" * 60)
    print("Helios++ Mobile Laser Scanning Simulation")
    print("Human with Heron Lite Backpack Scanner")
    print("=" * 60)
    
    # Configuration file paths
    survey_file = "survey.xml"
    
    # Output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if all required files exist
    required_files = [
        survey_file,
        "building1.obj", "building1.mtl",
        "building2.obj", "building2.mtl",
        "obstacle1.obj", "obstacle1.mtl",
        "obstacle2.obj", "obstacle2.mtl",
        "trajectory.txt"
    ]
    
    print("\nChecking required files...")
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"  ❌ {file} - NOT FOUND")
        else:
            print(f"  ✓ {file} - OK")
    
    if missing_files:
        print(f"\nError: Missing {len(missing_files)} required files!")
        return
    
    try:
        # Initialize Helios++ simulation
        print("\nInitializing Helios++ simulation...")
        
        # Assets paths as list (required by pyhelios)
        assets_paths = [
            ".",  # Current directory
            "assets/"  # Assets directory
        ]
        
        # Create simulation context with correct constructor signature
        sim = pyhelios.Simulation(
            survey_file,        # Survey file path
            assets_paths,       # Assets paths as list
            output_dir,         # Output directory
            1,                  # Number of threads
            False,              # Write waveform
            False,              # Write pulse
            False,              # Calculate echo width
            False               # Full wave noise
        )
        
        # Simulation statistics before run
        print("\nSimulation Configuration:")
        print(f"  - Scanner: Heron Lite (32 channels)")
        print(f"  - Platform: Human with backpack")
        print(f"  - Walking speed: 0.8 m/s")
        print(f"  - Total distance: 502.96 m")
        print(f"  - Estimated duration: 10.48 minutes")
        print(f"  - Scan frequency: 24 Hz")
        print(f"  - Max range: 120 m")
        
        # Start simulation
        print("\nStarting simulation...")
        start_time = time.time()
        
        # Run the simulation
        sim.start()
        
        # Wait for completion - simple approach
        print("Simulation running... (this may take several minutes)")
        while sim.isRunning():
            time.sleep(2.0)
            print(".", end="", flush=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n\nSimulation completed in {execution_time:.2f} seconds")
        
        # Check for output files
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            if output_files:
                print(f"\nGenerated files in '{output_dir}':")
                for file in output_files:
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"  - {file} ({file_size:.2f} MB)")
            else:
                print(f"\nNo output files found in '{output_dir}'")
        
        print("\n✓ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n\n❌ Error during simulation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Check that all file paths are correct")
        print("  2. Ensure Helios++ is properly installed")
        print("  3. Verify that the XML files are valid")
        print("  4. Check that .obj/.mtl files are in the correct format")
        print("  5. Make sure pyhelios bindings are compatible with your Helios++ version")
        return
    
    print("\n" + "=" * 60)

def main():
    """Main function"""
    # Optional: Set Helios++ verbosity level
    # 0 = silent, 1 = normal, 2 = verbose
    os.environ['HELIOS_VERBOSITY'] = '1'
    
    # Run the simulation
    run_simulation()

if __name__ == "__main__":
    main()