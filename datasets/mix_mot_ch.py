import argparse
import os
import yaml
from pathlib import Path


datasets = ['crowdhuman', 'MOT17', 'MOT20']


def prepare_yaml(out_path):
    data = {
        'path': str(out_path),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["person"]
    }
    with open(out_path / "dataset.yaml", "w") as f:
        yaml.dump(data, f)

def main(paths, out_path, splits=['train', 'val'], debug=False):
    os.makedirs(out_path, exist_ok=True)
    for split in splits:
        out_path_split = os.path.join(out_path, split)
        out_path_split_images = os.path.join(out_path_split, 'images')
        out_path_split_labels = os.path.join(out_path_split, 'labels')
        os.makedirs(out_path_split_images, exist_ok=True)
        os.makedirs(out_path_split_labels, exist_ok=True)

        for dataset in datasets:

            path = os.path.join(paths, dataset, 'yolo')
            assert os.path.exists(path), f"Path {path} does not exist"

            folders = os.listdir(path)
            data_dir = [f for f in folders if split in f][0]

            print(f"Working on {dataset} and {data_dir}")

            image_dir = os.path.join(data_dir, 'images')
            label_dir = os.path.join(data_dir, 'labels')

            image_files = os.listdir(os.path.join(path, image_dir))
            label_files = os.listdir(os.path.join(path, label_dir))
            print(f'Working on {image_dir} and {label_dir}')
            print(f"Found {len(image_files)} images and {len(label_files)} labels")
            print(f"Copying {data_dir} from {dataset} to {out_path_split}")
            for image_file in image_files:
                if os.path.exists(os.path.join(out_path_split_images, image_file)):
                    continue
                image_path = os.path.join(path, image_dir, image_file)
                out_image_path = os.path.join(out_path_split_images, image_file)
                os.symlink(image_path, out_image_path)

            for label_file in label_files:
                if os.path.exists(os.path.join(out_path_split_labels, label_file)):
                    continue
                label_path = os.path.join(path, label_dir, label_file)
                out_label_path = os.path.join(out_path_split_labels, label_file)
                os.symlink(label_path, out_label_path)
    prepare_yaml(Path(out_path))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--paths", type=str, help="paths to yolo datasets")
    args.add_argument("--out_path", type=str, help="path to output directory")

    args = args.parse_args()
    main(args.paths, args.out_path)
