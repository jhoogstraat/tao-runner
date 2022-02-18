import subprocess
from shutil import rmtree

from ..context import ExperimentContext


def run(context: ExperimentContext, overwrite: bool = False, stop: bool = False, **kwargs):
    # Checks to make sure all files are present and we don't override anything
    assert context.local_paths.train_spec_file.is_file(
    ), f"Spec file is not present at location '{context.local_paths.train_spec_file}'"
    if not overwrite:
        assert not context.local_paths.model_dir.exists(
        ), f"The model directory '{context.local_paths.model_dir.name}' already exists."
    elif context.local_paths.model_dir.exists():
        rmtree(context.local_paths.model_dir)

    # Prepare the model directory
    context.local_paths.model_dir.mkdir(exist_ok=True)

    with open(context.local_paths.train_spec_file, 'r') as infile, open(context.local_paths.model_dir.joinpath(context.local_paths.train_spec_file.name), 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$project", context.project)
        spec = spec.replace(
            "$dataset", context.docker_paths.dataset_dir.as_posix())
        spec = spec.replace(
            "$tfrecords", context.docker_paths.subset_tfrecords_dir.as_posix())
        spec = spec.replace("$pretrained_model",
                            context.docker_paths.pretrained_model_file.as_posix())
        outfile.write(spec)

    # Stop running training, if requested
    if stop:
        print("Stopping running tao tasks...")
        subprocess.run(["tao", "stop", "--all"], check=True,
                       text=True, capture_output=True)

    print("Starting training...")
    print(
        f"Using pretrained model: {context.config.repository}/{context.docker_paths.pretrained_model_file.name}")
    log_file = context.local_paths.model_dir.joinpath("train.log").as_posix()
    print(f"See {log_file} for training progress")

    completed = subprocess.run(["tao", context.config.head, "train",
                                "--gpus", context.tao.gpus,
                                "--gpu_index", context.tao.gpu_indices,
                                "-e", context.docker_paths.model_dir.joinpath(
                                    context.local_paths.train_spec_file.name).as_posix(),
                                "-r", context.docker_paths.model_dir.as_posix(),
                                "-k", context.config.model_key,
                                "--log_file", context.docker_paths.model_dir.joinpath("train.log").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)
