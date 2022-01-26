# Disclaimer: This is work in progress!

# Setup
  1. Install the required python packages (./scripts/setup_venv.sh can help you with that).  
  2. Change the paths in `.tao_mounts.json` according to your local system.

The main entry point is `python scripts/run.py`

# Configuration

## Folder Structure
A strict folder structure is required.

projects/  
├─ example_01/  
│  ├─ data/  
│  │  ├─ kitti_detection/  
│  │  ├─ tfrecords/  
│  │  ├─ vott_json/  
│  ├─ models/  
│  │  ├─ experiment_01/  
│  │  ├─ experiment_02/  
│  ├─ specs/  
│  │  ├─ experiment_01/  
│  │  ├─ experiment_02/  
│  ├─ experiments.cfg  
├─ example_02/  
│  ├─ ...

## Mounts into the tao container:
 1. experiments/ --> /workspace/experiments
 2. repositories/ --> /workspace/repositories

## Available overrides for tao config files:
 - $experiment: Name of the experiment (section in experiments.cfg)
 - $dataset: Path to the 'data' directory in the configured experiment (docker side).
 - $pretrained_model: Path to the pretrained model file (.hdf5)

# Converting a Dataset
Use `convert_dataset.sh` to convert between different data formats.
TAO mostly uses the KITTI format for object detection.  
After that, you can use 'python scripts/run.py convert -h' to see how to convert a KITTI dataset to TFRecords.

# Running a Training
I recommend using the [samples](https://api.ngc.nvidia.com/v2/resources/nvidia/tao/cv_samples/versions/v1.3.0/zip) from NVIDIA as a starting point.  
See `python scripts/run.py train -h` for the required arguments.  
You always provide the project (-p) and experiment (-e) when using the script and the task you want to carry out (convert, train or export).

Examples:  
- `python scripts/run.py -p example_01 -e experiment_01 convert`  
- `python scripts/run.py -p example_01 -e experiment_02 train`