from pathlib import Path
import random
from re import T
from shutil import rmtree, copyfile
from typing import List
from ..context import ExperimentContext
from .convert import check_kitti


def copy_kitti_set(src: Path, dest: Path, img_set: List[Path]):
    for img_src in img_set:
        label_src = src.joinpath("label_2", img_src.stem + ".txt")

        img_dest = dest.joinpath("image_2", img_src.name)
        label_dest = dest.joinpath("label_2", label_src.name)

        copyfile(img_src, img_dest)
        copyfile(label_src, label_dest)


def run(context: ExperimentContext, subset: str, val: float = 0.2, overwrite: bool = False, **kwargs):
    """Split a KITTI Dataset into train / val subsets."""
    print(f"Splitting subset {subset} into train and val subsets")

    dataset = context.local_paths.dataset_dir.joinpath(subset)

    check_kitti(dataset)

    if context.local_paths.subset_train_dir.exists():
        assert overwrite, f"The directory '{context.local_paths.subset_train_dir.name}' already exists at 'data/{context.local_paths.dataset_dir.name}'. Use --overwrite to replace the existing data."
        rmtree(context.local_paths.subset_train_dir)

    if context.local_paths.subset_val_dir.exists():
        assert overwrite, f"The directory '{context.local_paths.subset_val_dir.name}' already exists at 'data/{context.local_paths.dataset_dir.name}/'. Use --overwrite to replace the existing data."
        rmtree(context.local_paths.subset_val_dir)

    context.local_paths.subset_train_dir.joinpath(
        "image_2").mkdir(parents=True, exist_ok=True)
    context.local_paths.subset_train_dir.joinpath(
        "label_2").mkdir(parents=True, exist_ok=True)

    context.local_paths.subset_val_dir.joinpath(
        "image_2").mkdir(parents=True, exist_ok=True)
    context.local_paths.subset_val_dir.joinpath(
        "label_2").mkdir(parents=True, exist_ok=True)

    images_dir = dataset.joinpath("image_2")
    labels_dir = dataset.joinpath("label_2")

    images = list(
        images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob("*.jpeg"))

    total_cnt = len(images)
    val_cnt = int(total_cnt * val)
    train_cnt = total_cnt - val_cnt

    random.shuffle(images)

    train_img = images[:train_cnt]
    val_img = images[train_cnt:]

    print("Total {} samples in KITTI training dataset".format(total_cnt))
    print("{} for train and {} for val".format(train_cnt, val_cnt))
    print("Copying...")

    copy_kitti_set(dataset,
                   context.local_paths.subset_train_dir, train_img)
    copy_kitti_set(dataset,
                   context.local_paths.subset_val_dir, val_img)
