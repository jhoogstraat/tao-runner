from genericpath import exists
from shutil import copyfile, rmtree
import argparse
import configparser
from os import environ
from pathlib import Path
import subprocess

# Parse Arguments

parser = argparse.ArgumentParser(
    description='Run different tao commands according to your training.cfg')
subparsers = parser.add_subparsers(help='sub-command help', dest='command')

# Top-level parser
parser.add_argument('-e', '--experiment',
                    help='The name of the experiment (required)', required=True)
parser.add_argument('-t', '--training',
                    help='The section of your training.cfg to use (required)', required=True)

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

# Loading config
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(Path("experiments", args.experiment, "training.cfg"))

EXPERIMENT = args.experiment
TRAINING = args.training
HEAD = config.get(TRAINING, "HEAD")
BACKBONE = config.get(TRAINING, "BACKBONE")
REPOSITORY = config.get(TRAINING, "REPOSITORY")
SPEC = config.get(TRAINING, "SPEC")
MODEL_KEY = config.get(TRAINING, "MODEL_KEY")

# Set local dirs and files according to the Config
LOCAL_EXPERIMENT_DIR = Path.cwd().joinpath("experiments", EXPERIMENT)
LOCAL_DATA_DIR = LOCAL_EXPERIMENT_DIR.joinpath("data")
LOCAL_MODEL_DIR = LOCAL_EXPERIMENT_DIR.joinpath("models", TRAINING)
LOCAL_SPEC_FILE = LOCAL_EXPERIMENT_DIR.joinpath("specs", HEAD, SPEC)

# Set docker dirs and files accoring to training.cfg (and .tao_mounts.json)
DOCKER_EXPERIMENT_DIR = Path("/workspace/experiments/", EXPERIMENT)
DOCKER_DATA_DIR = DOCKER_EXPERIMENT_DIR.joinpath("data")
DOCKER_MODEL_DIR = DOCKER_EXPERIMENT_DIR.joinpath("models", TRAINING)
DOCKER_SPEC_FILE = DOCKER_MODEL_DIR.joinpath("spec.cfg")
DOCKER_PRETRAINED_MODEL_FILE = Path(
    "/repositories/", REPOSITORY, REPOSITORY + "v" + BACKBONE + ".hdf5")


def train():
    # Checks to make sure all files are present and we don't override anything
    assert LOCAL_SPEC_FILE.is_file(
    ), f"Spec file is not present at location '{SPEC}'"
    if not args.overwrite:
        assert not LOCAL_MODEL_DIR.exists(
        ), f"The model directory '{TRAINING}' already exists."
    elif LOCAL_MODEL_DIR.exists():
        rmtree(LOCAL_MODEL_DIR)

    # Prepare the tao training config
    # Copies the specified spec to the MODEL
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)
    with open(LOCAL_SPEC_FILE, 'r') as infile, open(LOCAL_MODEL_DIR.joinpath("spec.cfg"), 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$dataset", DOCKER_DATA_DIR.as_posix())
        outfile.write(spec)

    # TAO uses ~/.tao_mounts.json, so copying the file there...
    copyfile(Path(".tao_mounts.json"),
             Path.home().joinpath(".tao_mounts.json"))

    # Temporary fix for the TAO Docker Image using WSL 2.
    # https://forums.developer.nvidia.com/t/wsl2-tao-issues/195476
    environ["OVERRIDE_REGISTRY"] = "local.pwn"

    # Stop running training, if requested
    if args.stop:
        print("Stopping running tao tasks...")
        subprocess.run(["tao", "stop", "--all"], check=True,
                       text=True, capture_output=True)

    print("Starting training...")
    log_file = LOCAL_MODEL_DIR.joinpath("train.log").as_posix()
    print(f"See {log_file} for training progress")

    completed = subprocess.run(["tao", HEAD, "train",
                                "--gpus", config.get("BASE", "GPUS"),
                                "--gpu_index", config.get("BASE", "GPU_INDEX"),
                                "-e", DOCKER_SPEC_FILE.as_posix(),
                                "-r", DOCKER_MODEL_DIR.as_posix(),
                                "-k", MODEL_KEY,
                                "--gen_ds_config"
                                "--log_file", DOCKER_MODEL_DIR.joinpath("train.log").as_posix()], check=True, text=True, capture_output=True)

    print(completed.stdout)


def export():
    model_name = config.get(TRAINING, "EXPORT_MODEL")
    local_model_path = Path(LOCAL_MODEL_DIR.joinpath(
        "weights", model_name + ".tlt"))
    assert local_model_path.exists(
    ), f"Model at {local_model_path} does not exist"

    print(
        f"Exporting {model_name}, experiment: {EXPERIMENT}, training: {TRAINING}")
    log_file = LOCAL_MODEL_DIR.joinpath("export.log").as_posix()
    print(f"See {log_file} for training progress")

    data_type = config.get(TRAINING, "EXPORT_TYPE")
    completed = subprocess.run(["tao", HEAD, "export",
                                "-m", DOCKER_MODEL_DIR.joinpath(
                                    "weights", model_name + ".tlt").as_posix(),
                                "-k", MODEL_KEY,
                                "-e", DOCKER_SPEC_FILE.as_posix(),
                                "-o", DOCKER_MODEL_DIR.joinpath(
                                    "export", f"f{model_name}_{data_type}.etlt").as_posix(),
                                "--data_type", data_type,
                                "--gen_ds_config",
                                "--gpu_index", config.get("BASE", "GPU_INDEX"),
                                "--log_file", DOCKER_MODEL_DIR.joinpath("export.log").as_posix()], check=True, text=True, capture_output=True)

    print(completed.stdout)


if args.command == "train":
    train()
elif args.command == "export":
    export()
