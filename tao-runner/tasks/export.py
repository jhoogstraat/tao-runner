import subprocess
from pathlib import Path

from ..context import ExperimentContext


def run(context: ExperimentContext, **kwargs):
    model_name = context.config.export_model
    local_model_path = Path(context.local_paths.model_dir.joinpath(
        "weights", model_name + ".tlt"))
    assert local_model_path.exists(
    ), f"Model at {local_model_path} does not exist"

    print(
        f"Exporting {model_name}, project: {context.project}, experiment: {context.experiment}")
    log_file = context.local_paths.model_dir.joinpath("export.log").as_posix()
    print(f"See {log_file} for training progress")

    data_type = context.config.export_type
    completed = subprocess.run(["tao", context.config.head, "export",
                                "-m", context.docker_paths.model_dir.joinpath(
                                    "weights", model_name + ".tlt").as_posix(),
                                "-k", context.config.model_key,
                                "-e", context.docker_paths.specs_dir.as_posix(),
                                "-o", context.docker_paths.model_dir.joinpath(
                                    "export", f"{model_name}_{data_type}.etlt").as_posix(),
                                "--data_type", data_type,
                                "--gen_ds_config",
                                "--gpu_index", context.tao.gpus_indices,
                                "--log_file", context.docker_paths.model_dir.joinpath("export.log").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)

    # -o Seems to be dependent on the Model. DSSD: NMS
    # d is the dimension of the input. Model dependent!
    tao_converter_command = "tao-converter \\\n" + \
                            f"\t\t-d 3,300,300\\\n" + \
                            f"\t\t-o NMS \\\n" + \
                            f"\t\t-k {context.config.model_key} \\\n" + \
                            f"\t\t-e export/{model_name}_{data_type}.fp16.engine \\\n" + \
                            f"\t\t-t {data_type} \\\n" + \
                            f"\t\t-m 1 \\\n" + \
                            f"\t\texport/{model_name}_{data_type}.etlt"

    with open(context.local_paths.model_dir.joinpath("generate_trt_engine.sh"), 'w') as outfile:
        outfile.write(tao_converter_command)
