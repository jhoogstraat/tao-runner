# Disclaimer: This is work in progress!

# Setup
Install the required python packages (./scripts/setup_venv.sh can help you witht that).  
Set the paths in `.tao_mounts.json` to your local system.

The main entry point is `python scripts/run.py`

# Configure the training

## Folder Structure
A strict folder structure is required.

experiments/  
├─ example_01/  
│  ├─ data/  
│  ├─ models/  
│  ├─ specs/  
│  ├─ training.cfg  
├─ example_02/  
│  ├─ ...

## Mounts into the tao container:
 1. experiments/ --> /workspace/experiments
 2. repositories/ --> /workspace/repositories

## Available overrides in tao config files:
 - $dataset: Path to the 'data' directory in the configured experiment (docker side).

# Running a Training
I recommend to use the [samples](https://api.ngc.nvidia.com/v2/resources/nvidia/tao/cv_samples/versions/v1.3.0/zip) from nvidia as a starting point.

## Convert Dataset
Use `convert_dataset.sh` to convert between different data formats.
TAO mostly uses the KITTI format for object detection.

## Starting
See `python scripts/run.py train -h` for the required arguments.