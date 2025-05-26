#!/usr/bin/env python3
"""
Alternative: Run Helios++ simulation using command-line interface
This is simpler if you have issues with Python bindings
"""

import subprocess
import os
import sys
import time

def run_simulation_cli():
    """Run Helios++ using command line interface"""
    
    print("=" * 60)
    print("Helios++ Mobile Laser Scanning Simulation (CLI)")
    print("Human with Heron Lite Backpack Scanner")
    print("=" * 60)
    
    # Helios++ executable path - adjust this to your installation
    helios_exe = "helios"  # or full path like "/path/to/helios++"
    
    # Check if helios executable exists
    try:
        result = subprocess.run([helios_exe, "--help"], 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            raise Exception("Helios++ not working properly")
    except:
        print(f"\nError: Cannot find or run '{helios_exe}'")
        print("\nPlease either:")
        print("  1. Add Helios++ to your PATH, or")
        print("  2. Update the 'helios_exe' variable with the full path")
        print("  3. Make sure Helios++ is properly installed")
        return
    
    # Survey file
    survey_file = "survey.xml"
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build command
    cmd = [
        helios_exe,
        survey_file,
        "--output", output_dir,
        "--las",  # Output in LAS format
        "--splitByChannel",  # Split output by channel
        "--seed", "42",  # Random seed for reproducibility
        "--threads", "4"  # Number of threads (adjust as needed)
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nStarting simulation...")
    print("-" * 40)
    
    try:
        # Run simulation
        start_time = time.time()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        process.wait()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if process.returncode == 0:
            print("-" * 40)
            print(f"\n✓ Simulation completed successfully!")
            print(f"  Execution time: {execution_time:.2f} seconds")
            
            # List output files
            output_files = os.listdir(output_dir)
            if output_files:
                print(f"\nGenerated files in '{output_dir}':")
                for file in output_files:
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  - {file} ({file_size:.2f} MB)")
        else:
            print(f"\n❌ Simulation failed with return code: {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_simulation_cli()
