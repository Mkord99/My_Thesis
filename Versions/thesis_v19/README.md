# Visibility Path Planning

A modular implementation for path planning with visibility constraints. This program generates optimal paths that can observe building boundaries while avoiding obstacles.

## Directory Structure

```
visibility_path_planning/
│
├── config/                         # Configuration files
│   └── config.json                 # Main configuration file
│
├── data/                           # Data storage
│   └── geometry.json               # Building layouts and obstacles
│
├── output/                         # Output files
│   ├── logs/                       # Log files 
│   └── plots/                      # Plot images
│
├── src/                            # Source code
│   ├── __init__.py                 # Package marker
│   ├── data_handler.py             # Geometry loading/processing
│   ├── graph_builder.py            # Graph construction
│   ├── visibility_analyzer.py      # Visibility calculations
│   ├── optimizer.py                # Optimization model
│   ├── visualizer.py               # Plotting functions
│   └── utils.py                    # Helper functions
│
├── main.py                         # Entry point
└── README.md                       # Documentation
```

## Requirements

- Python 3.8+
- Required packages:
  - networkx
  - numpy
  - matplotlib
  - shapely
  - scipy
  - gurobipy (requires a license)

## Configuration

The program can be configured using the `config.json` file. Key configuration options include:

- Graph parameters (grid spacing, edge distances)
- Visibility constraints (angles, distances)
- Optimization settings (VRF weighting, tie points)
- Output options (logging, plot generation)

## Building and Obstacle Data

Building and obstacle geometries are defined in the `geometry.json` file. This includes:

- Buildings: Polygon coordinates
- Obstacles: Polygon coordinates with type (radiation, visibility)

## Usage

1. Configure the `config.json` file to set parameters
2. Define buildings and obstacles in `geometry.json`
3. Run the program:

```bash
python main.py
```

## Features

- Modular design for easy maintenance and extension
- Configurable parameters for different use cases
- Visibility analysis with particle-based approach
- Optimization with visibility ratio factor (VRF) weighting
- Comprehensive visualization of results
- Detailed logging of process steps

## Output

- Plot image of the optimized path
- Log file with execution details