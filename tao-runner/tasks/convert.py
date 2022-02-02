import subprocess
from pathlib import Path
from shutil import rmtree

from ..context import ExperimentContext


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
                lines = j.readlines()

                # Check if no labels are associated with this image. Removing image + label in that case.
                if not lines:
                    image_file = list(images_dir.glob(label_file.stem + "*"))
                    if len(image_file) != 1:
                        print(
                            f"Found {len(image_file)} matching images to label file {label_file}. Expected 1.")
                        continue
                    else:
                        print(
                            f"Removing image and label for sample {label_file.stem}")
                        image_file[0].unlink()
                        label_file.unlink()

                # Check if each row contains 15 columns. Remove the 16th if necessary.
                for line in lines:
                    row = line.strip().split(" ")
                    if len(row) > 15:
                        print(
                            f"The file {label_file.name} has {len(row)} fields. Saving first 15")
                        new_label = str.join(" ", row[:15])
                        with open(label_file, 'w') as w:
                            w.write(new_label)
    print(f"Checked labels for {len(samples)} image files. All good.")


def run(context: ExperimentContext, overwrite: bool = False, **kwargs):
    assert context.local_paths.convert_spec_file.is_file(
    ), f"Converter Spec file does not exist at location '{context.local_paths.convert_spec_file}'"

    if context.local_paths.data_tfrecords_dir.exists():
        assert overwrite, f"The directory '{context.local_paths.data_tfrecords_dir.name}' already exists at 'data/'. Use --overwrite to replace the existing data."
        rmtree(context.local_paths.data_tfrecords_dir)
    else:
        context.local_paths.data_tfrecords_dir.mkdir()

    with open(context.local_paths.convert_spec_file, 'r') as infile, open(context.local_paths.compiled_convert_spec_file, 'w') as outfile:
        spec = infile.read()
        spec = spec.replace("$project", context.project)
        spec = spec.replace(
            "$dataset_raw", context.docker_paths.data_raw_dir.as_posix())
        spec = spec.replace(
            "$dataset_tfrecord", context.docker_paths.data_tfrecords_dir.as_posix())
        spec = spec.replace("$pretrained_model",
                            context.docker_paths.pretrained_model_file.as_posix())
        outfile.write(spec)

    check_kitti(context.local_paths.data_raw_dir)

    print("Converting dataset to TFRecords...")
    completed = subprocess.run(["tao", context.config.head, "dataset_convert",
                                "-d", context.docker_paths.compiled_convert_spec_file.as_posix(),
                                "-o", context.docker_paths.data_tfrecords_dir.joinpath("tfrecord").as_posix()], check=False, text=True, capture_output=True)

    print("STDOUT:", completed.stdout)
    print("STDERR:", completed.stderr)
