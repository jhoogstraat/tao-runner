import subprocess
from pathlib import Path
from shutil import rmtree

from ..context import ExperimentContext


# Datumaro does not export label files for 'background' images, which do not contain any object to be detected.
# TAO expects kitti label files to have 15 columns, but kitti has 16 originally (https://github.com/NVIDIA/DIGITS/issues/992)


def check_kitti(kitti_dir: Path):
    assert kitti_dir.exists(), f"The directory {kitti_dir} does not exist"

    images_dir = kitti_dir.joinpath("image_2")
    labels_dir = kitti_dir.joinpath("label_2")
    images = list(
        images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob("*.jpeg"))
    labels = list(labels_dir.glob('*.txt'))

    assert len(images) > 0, f"No samples found in dataset {kitti_dir}"

    if len(labels) <= len(images):
        missing_files = [file.stem for file in labels if file not in images]
        assert f"Found {len(labels)} label files, but only {len(images)} images. Missing:\n" + \
            '\n'.join(missing_files)

    print(f"Images: {len(images)}, labels: {len(labels)}")

    for image in images:
        label_file = labels_dir.joinpath(image.with_suffix('.txt').name)
        if not label_file.exists():
            with open(label_file, 'w') as r:
                r.write('')
                print(f"Created empty label file {label_file.name}")
        else:
            with open(label_file, 'r') as r:
                for line in r.readlines():
                    row = line.strip().split(" ")
                    if len(row) > 15:
                        print(
                            f"The file {label_file.name} has {len(row)} fields. Saving first 15")
                        new_label = str.join(" ", row[:15])
                        with open(label_file, 'w') as w:
                            w.write(new_label)
    print(f"Checked labels for {len(images)} images. All good.")


def run(context: ExperimentContext, overwrite: bool = False, **kwargs):
    assert context.local_paths.convert_spec_file.is_file(
    ), f"Converter spec file does not exist at location '{context.local_paths.convert_spec_file}'"

    # TODO: We currently don't know which subset (full, train, val, ...) is being converted, so we cannot check if the dataset is ok.
    # check_kitti(dataset)

    if context.local_paths.subset_tfrecords_dir.exists():
        assert overwrite, f"The directory '{context.local_paths.subset_tfrecords_dir.name}' already exists at 'data/'. Use --overwrite to replace the existing data."
        rmtree(context.local_paths.subset_tfrecords_dir)

    context.local_paths.subset_tfrecords_dir.mkdir()

    with open(context.local_paths.convert_spec_file, 'r') as infile, open(context.local_paths.compiled_convert_spec_file, 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$project", context.project)
        spec = spec.replace(
            "$dataset", context.docker_paths.dataset_dir.as_posix())
        spec = spec.replace(
            "$dataset_tfrecord", context.docker_paths.subset_tfrecords_dir.as_posix())
        spec = spec.replace("$pretrained_model",
                            context.docker_paths.pretrained_model_file.as_posix())
        outfile.write(spec)

    print("Converting dataset to TFRecords...\n")
    completed = subprocess.run(["tao", context.config.head, "dataset_convert",
                                "-d", context.docker_paths.compiled_convert_spec_file.as_posix(),
                                "-o", context.docker_paths.subset_tfrecords_dir.joinpath("tfrecord").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)
