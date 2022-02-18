# What is Tao-Runner
Tao-Runner is a opinionated way to organize and train machine learning models with NVIDIA TAO. It does all the housekeeping around directory structures and keeping your experiments clean.

# Setup
  1. Clone this repository to a suitable directory for your projects and create a 'projects/' dir inside the cloned repo.
  2. Install the required python packages, preferably in a virtual environment (./scripts/setup_venv.sh can help you with that).  
  3. Change the paths in `.tao_mounts.json` according to your local system.
  4. Download the pretrained models from NVIDIA NGC (using `python scripts/download_pretrained_models.py`)

**The main entry point is `python -m tao-runner`**

# Configuration

## Folder Structure
A strict folder structure is required and enforced by `tao-runner`.

projects/  
├─ example_01/  
│  ├─ data/  
│  │  ├─ kitti_detection/  
│  │  ├─ vott_json/  
│  ├─ models/  
│  │  ├─ experiment_01/  
│  │  ├─ experiment_02/  
│  ├─ specs/  
│  │  ├─ experiment_01/  
│  │  ├─ experiment_02/  
│  ├─ experiments.yml  
├─ example_02/  
│  ├─ ...

## Datasets
Each dataset is referenced by its dir name under data/.  
A dataset can contain multiple subsets, such as `full`, `train` and `val`.  
You can name your subsets however you want, as long as you reference them correctly in your tao config (e.g. `$dataset/full` or `$dataset/custom_subset`)
Most model architectures require just the `full` dataset containing all images.  
`retinanet` uses two datasets (`train` and `val`) which can be generated via the `split` task from the `full` set.

## Mounts into the tao container:
 1. experiments/ --> /workspace/experiments
 2. repositories/ --> /workspace/repositories

## Available overrides for tao config files:
 - `$experiment`: Name of the experiment (name of the section in experiments.cfg).
 - `$dataset`: Path to the dataset (most likely in kitti format) as configured in `experiments.yml` (docker side).
 - `$tfrecords`: Path to the tfrecord-formatted dataset directory (docker side).
 - `$pretrained_model`: Path to the pretrained model file (.hdf5 file, docker side).


# Tasks
## Converting a dataset
Use `convert_dataset.sh` to convert between different data formats.
TAO mostly uses the KITTI format for object detection.  
After that, you can use `python -m tao-runner convert -h` to see how to convert a KITTI dataset to TFRecords.
This task is idempotent.

Examples:  
- `python -m tao-runner convert example_01 experiment_01`  
- `python -m tao-runner convert example_01  experiment_01 experiment_02 --overwrite`

## Splitting a dataset
Use this task to split the dataset into disjunct `train` and `val` subset.
You can set the percentage of the `val` subset by setting `--val` to a value between `0.0` and `1.0`.
This task is idempotent.

Required by:
- RetinaNet

Examples:  
- `python -m tao-runner split --subset full --val 0.1 example_01 experiment_01`
- `python -m tao-runner split --subset custom_subset example_01 experiment_01 experiment_02 --overwrite`


## Running a Training
I recommend using the [samples](https://api.ngc.nvidia.com/v2/resources/nvidia/tao/cv_samples/versions/v1.3.0/zip) from NVIDIA as a starting point.  
See `python -m tao-runner train -h` for the required arguments.  
You always provide the task you want to carry out (`split`, `convert`, `train` or `export`), the project and the experiments for which the tasks are executed.
This task is idempotent.

Examples:  
- `python -m tao-runner train example_01 experiment_01`  
- `python -m tao-runner train example_01 experiment_01 experiment_02 --overwrite`


# experiments.yml
This file defines all your different experiments inside of a project.  
The following example explains its structure in detail:
```yaml
# This section contains tao-specific configurations.
tao_config:
  # Copied over from tao. Denominates the number of gpus to use.
  gpus: 1
  # Copied over from tao. List the indices of the gpus to use (see nvidia-smi)
  gpu_indices: [1]

# This sections contains all experiments inside the project.
# Create new named experiments via keys:
experiments:
  # The name of the experiment
  dssd_resnet18_01:
    # Detection head ("retinanet", "dssd", "detectnet_v2")
    head: dssd
    # Detection Backbone ("resnet10", "resnet18", "resnet50", "vgg16", "vgg19", etc.)
    backbone: resnet18
    # Directory of the repository which holds the pretrained model to be used (dir under 'repositories')
    repository: pretrained_object_detection
    # Encription key of the trained model
    model_key: secret_key
    # Directory of the raw dataset under 'data/' to use. Often the kitti_detection format is required.
    dataset: kitti_detection
    # Filename of the model to export (.tlt model)
    export_model: dssd_resnet18_epoch_080
    # The data type of the exported model (fp32, fp16, int8)
    export_type: fp16

  detectnetv2_resnet18_01:
    head: detectnet_v2
    backbone: resnet18
    repository: pretrained_detectnet_v2
    model_key: secret_key
    dataset: kitti_detection_1000x1000
    export_model: detectnet_v2_resnet18_epoch_080
    export_type: fp16
```

## Exporting a model
Todo....