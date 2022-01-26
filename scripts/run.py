import argparse
import configparser
import subprocess
from copy import copy
from os import environ
from pathlib import Path
from shutil import copyfile, rmtree

parser = argparse.ArgumentParser(
    description='Run different tao commands according to your experiments.cfg')
subparsers = parser.add_subparsers(help='sub-command help', dest='command')

# Top-level parser
parser.add_argument('-p', '--project',
                    help='The name of the project (required)', required=True)
parser.add_argument('-e', '--experiment',
                    help='The section of your experiments.cfg to use (required)', required=True)

parser_convert = subparsers.add_parser(
    'convert', help="Convert a dataset to tfrecords")
parser_convert.add_argument('--overwrite',
                            help='If this flag is set, the model dir will be completly removed and recreated', action="store_true")

# Parser for "train" command
parser_train = subparsers.add_parser('train', help='Train a model')
parser_train.add_argument(
    '-s', '--stop', help='Stop all running training sessions', action="store_true")
parser_train.add_argument('--overwrite',
                          help='If this flag is set, the model dir will be completly removed and recreated', action="store_true")

# Parser for "export" command
parser_export = subparsers.add_parser('export', help='Export a model')
parser_export.add_argument('-m', '--model',
                           help='The Filename of the model to export')
parser_export.add_argument(
    '-t', '--type', help='The desired engine data type', default='fp16')

# Load config
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(Path("projects", args.project, "experiments.cfg"))

# Extract configuration from config file.
PROJECT = args.project
EXPERIMENT = args.experiment
HEAD = config.get(EXPERIMENT, "HEAD")
BACKBONE = config.get(EXPERIMENT, "BACKBONE")
REPOSITORY = config.get(EXPERIMENT, "REPOSITORY")
MODEL_KEY = config.get(EXPERIMENT, "MODEL_KEY")

# Set local dirs and files according to the Config
LOCAL_PROJECT_DIR = Path("projects", PROJECT)
LOCAL_DATA_DIR = LOCAL_PROJECT_DIR.joinpath("data")
LOCAL_MODEL_DIR = LOCAL_PROJECT_DIR.joinpath("models", EXPERIMENT)
LOCAL_SPECS_DIR = LOCAL_PROJECT_DIR.joinpath("specs", EXPERIMENT)
LOCAL_PRETRAINED_MODEL_DIR = Path(
    "repositories", REPOSITORY, REPOSITORY + "_v" + BACKBONE)

LOCAL_CONVERT_SPEC_FILE = LOCAL_SPECS_DIR.joinpath("convert.txt")
LOCAL_TRAIN_SPEC_FILE = LOCAL_SPECS_DIR.joinpath("train.txt")
LOCAL_PRETRAINED_MODEL_FILE = next(LOCAL_PRETRAINED_MODEL_DIR.glob("*.hdf5"))

# Set docker dirs and files accoring to experiments.cfg (and .tao_mounts.json)
DOCKER_PROJECT_DIR = Path("/workspace", "projects/", PROJECT)
DOCKER_DATA_DIR = DOCKER_PROJECT_DIR.joinpath("data")
DOCKER_MODEL_DIR = DOCKER_PROJECT_DIR.joinpath("models", EXPERIMENT)
DOCKER_SPEC_FILE = DOCKER_MODEL_DIR.joinpath("spec.cfg")
DOCKER_PRETRAINED_MODEL_FILE = Path("/workspace",
                                    "repositories/", REPOSITORY, REPOSITORY + "_v" + BACKBONE, LOCAL_PRETRAINED_MODEL_FILE.name)

# Setup
# TAO uses ~/.tao_mounts.json, so copying the file there...
copyfile(Path(".tao_mounts.json"),
         Path.home().joinpath(".tao_mounts.json"))

# Temporary fix for the TAO Docker Image using WSL 2.
# https://forums.developer.nvidia.com/t/wsl2-tao-issues/195476
environ["OVERRIDE_REGISTRY"] = "local.pwn"

# Datumaro does not export label files for 'background' images, which do not contain any object to be detected.
# TAO expects kitti label files to have 15 columns, but kitti has 16 originally (https://github.com/NVIDIA/DIGITS/issues/992)


def check_kitti(kitti_dir: Path):
    images_dir = kitti_dir.joinpath("image_2")
    labels_dir = kitti_dir.joinpath("label_2")
    samples = list(images_dir.glob('*.jpg')) + \
        list(images_dir.glob('*.png')) + list(images_dir.glob("*.jpeg"))
    for sample in samples:
        label_file = labels_dir.joinpath(sample.with_suffix('.txt').name)
        if not label_file.exists():
            with open(label_file, 'w') as j:
                j.write('')
                print(f"Created empty label file {label_file.name}")
        else:
            with open(label_file, 'r') as j:
                for line in j.readlines():
                    label = line.strip().split(" ")
                    length = len(label)
                    if length > 15:
                        print(
                            f"The file {label_file.name} has {length} fields. Saving first 15")
                        new_label = str.join(" ", label[:15])
                        with open(label_file, 'w') as w:
                            w.write(new_label)
    print(f"Checked labels for {len(samples)} image files. All good.")


def convert():
    assert LOCAL_CONVERT_SPEC_FILE.is_file(
    ), f"Converter Spec file is not present at location '{LOCAL_CONVERT_SPEC_FILE}'"

    TFRECORDS_DIR_NAME = "tfrecords"
    LOCAL_TFRECORD_DIR = LOCAL_DATA_DIR.joinpath(TFRECORDS_DIR_NAME)
    if not args.overwrite:
        assert not LOCAL_TFRECORD_DIR.exists(
        ), f"The directory '{TFRECORDS_DIR_NAME}' already exists under 'data'."
    elif LOCAL_TFRECORD_DIR.exists():
        rmtree(LOCAL_TFRECORD_DIR)

    LOCAL_TFRECORD_DIR.mkdir(exist_ok=True)

    with open(LOCAL_CONVERT_SPEC_FILE, 'r') as infile, open(LOCAL_TFRECORD_DIR.joinpath("converter_spec.cfg"), 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$project", PROJECT)
        spec = spec.replace("$dataset", DOCKER_DATA_DIR.as_posix())
        spec = spec.replace("$pretrained_model",
                            DOCKER_PRETRAINED_MODEL_FILE.as_posix())
        outfile.write(spec)

    DOCKER_TFRECORD_DIR = DOCKER_DATA_DIR.joinpath(TFRECORDS_DIR_NAME)
    DOCKER_CONVERTER_SPEC_FILE = DOCKER_TFRECORD_DIR.joinpath(
        "converter_spec.cfg")

    check_kitti(LOCAL_DATA_DIR.joinpath("kitti_detection", "train"))

    completed = subprocess.run(["tao", HEAD, "dataset_convert",
                                "-d", DOCKER_CONVERTER_SPEC_FILE.as_posix(),
                                "-o", DOCKER_TFRECORD_DIR.joinpath("tfrecord").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)


def train():
    # Checks to make sure all files are present and we don't override anything
    assert LOCAL_TRAIN_SPEC_FILE.is_file(
    ), f"Spec file is not present at location '{LOCAL_TRAIN_SPEC_FILE}'"
    if not args.overwrite:
        assert not LOCAL_MODEL_DIR.exists(
        ), f"The model directory '{EXPERIMENT}' already exists."
    elif LOCAL_MODEL_DIR.exists():
        rmtree(LOCAL_MODEL_DIR)

    # Prepare the tao training config
    # Copies the specified spec to the MODEL
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)
    with open(LOCAL_TRAIN_SPEC_FILE, 'r') as infile, open(LOCAL_MODEL_DIR.joinpath("spec.cfg"), 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$project", PROJECT)
        spec = spec.replace("$dataset", DOCKER_DATA_DIR.as_posix())
        spec = spec.replace("$pretrained_model",
                            DOCKER_PRETRAINED_MODEL_FILE.as_posix())
        outfile.write(spec)

    # Stop running training, if requested
    if args.stop:
        print("Stopping running tao tasks...")
        subprocess.run(["tao", "stop", "--all"], check=True,
                       text=True, capture_output=True)

    print("Starting training...")
    print(
        f"Using pretrained model: {REPOSITORY}/{DOCKER_PRETRAINED_MODEL_FILE.name}")
    log_file = LOCAL_MODEL_DIR.joinpath("train.log").as_posix()
    print(f"See {log_file} for training progress")

    completed = subprocess.run(["tao", HEAD, "train",
                                "--gpus", config.get("BASE", "GPUS"),
                                "--gpu_index", config.get("BASE", "GPU_INDEX"),
                                "-e", DOCKER_SPEC_FILE.as_posix(),
                                "-r", DOCKER_MODEL_DIR.as_posix(),
                                "-k", MODEL_KEY,
                                "--log_file", DOCKER_MODEL_DIR.joinpath("train.log").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)


def export():
    model_name = config.get(EXPERIMENT, "EXPORT_MODEL")
    local_model_path = Path(LOCAL_MODEL_DIR.joinpath(
        "weights", model_name + ".tlt"))
    assert local_model_path.exists(
    ), f"Model at {local_model_path} does not exist"

    print(
        f"Exporting {model_name}, project: {PROJECT}, experiment: {EXPERIMENT}")
    log_file = LOCAL_MODEL_DIR.joinpath("export.log").as_posix()
    print(f"See {log_file} for training progress")

    data_type = config.get(EXPERIMENT, "EXPORT_TYPE")
    completed = subprocess.run(["tao", HEAD, "export",
                                "-m", DOCKER_MODEL_DIR.joinpath(
                                    "weights", model_name + ".tlt").as_posix(),
                                "-k", MODEL_KEY,
                                "-e", DOCKER_SPEC_FILE.as_posix(),
                                "-o", DOCKER_MODEL_DIR.joinpath(
                                    "export", f"{model_name}_{data_type}.etlt").as_posix(),
                                "--data_type", data_type,
                                "--gen_ds_config",
                                "--gpu_index", config.get("BASE", "GPU_INDEX"),
                                "--log_file", DOCKER_MODEL_DIR.joinpath("export.log").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)

    # -o Seems to be dependent on the Model. DSSD: NMS
    # d is the dimension of the input. Model dependent!
    tao_converter_command = "tao-converter \\\n" + \
                            f"\t\t-d 3,300,300\\\n" + \
                            f"\t\t-o NMS \\\n" + \
                            f"\t\t-k {MODEL_KEY} \\\n" + \
                            f"\t\t-e export/{model_name}_{data_type}.fp16.engine \\\n" + \
                            f"\t\t-t {args.type} \\\n" + \
                            f"\t\t-m 1 \\\n" + \
                            f"\t\texport/{model_name}_{data_type}.etlt"

    with open(LOCAL_MODEL_DIR.joinpath("generate_trt_engine.sh"), 'w') as outfile:
        outfile.write(tao_converter_command)


# Run command
if args.command == "convert":
    convert()
elif args.command == "train":
    train()
elif args.command == "export":
    export()
