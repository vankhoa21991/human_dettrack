import os
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import shutil
import yaml
from multiprocessing import Pool
from itertools import repeat

class CocoToYoloCrowdHuman():
    def __init__(self, data_path: Path, out_path: Path=None, split="val", image_path=None, num_workers=8):
        self.data_path = data_path
        if out_path is None:
            self.out_path = data_path / "yolo"
        else:
            self.out_path = out_path
        self.splits = split
        self.path_to_img = image_path
        self.num_workers = num_workers

    def init_paths(self, split):
        self.path_to_annotations = self.data_path / "annotations" / f"{split}.json"

        if self.path_to_img is None:
            self.path_to_images = self.data_path / split
        else:
            self.path_to_images = Path(self.path_to_img)

        self.label_path = self.data_path / "yolo" / split / "labels"
        self.image_path = self.data_path / "yolo" / split / "images"

        self.file_names = []
        os.makedirs(self.label_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

    def load_coco_json(self):
        f = open(self.path_to_annotations)
        self.data = json.load(f)
        f.close()

    def prepare_images(self):
        print("Preparing images...")
        for img in tqdm(self.data['images']):
            filename = img['file_name']

            if os.path.exists(self.image_path / self.make_filename(img['file_name'])):
                self.file_names.append(filename)
                continue

            os.symlink(self.path_to_images / filename, self.image_path / self.make_filename(filename))
            self.file_names.append(filename)
        print("Done!")
    def run(self):
        for split in self.splits:
            self.init_paths(split)
            self.load_coco_json()
            self.prepare_images()
            self.prepare_labels()
        self.prepare_yaml()

    def get_img_ann(self, image_id):
        img_ann = []
        isFound = False
        for ann in self.data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        if isFound:
            return img_ann
        else:
            return None

    def get_img(self, filename):
        for img in self.data['images']:
            if filename in img['file_name']:
                return img
    def make_filename(self, filename):
        return filename.split("/")[-1]
    def prepare_label(self, filename):
        if os.path.exists(f"{self.label_path}/{self.make_filename(filename).replace('jpg', 'txt')}"):
            return

        # Extracting image
        img = self.get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Get Annotations for this image
        img_ann = self.get_img_ann(img_id)
        if img_ann:
            # Opening file for current image
            file_object = open(f"{self.label_path}/{self.make_filename(filename).replace('jpg', 'txt')}", "a")

            for ann in img_ann:
                current_category = ann['category_id'] - 1  # As yolo format labels start from 0
                current_bbox = ann['bbox']
                x = current_bbox[0]
                y = current_bbox[1]
                w = current_bbox[2]
                h = current_bbox[3]

                # round x, y, w, h
                x = max(int(x), 0)
                y = max(int(y), 0)
                w = min(int(w), img_w - x)
                h = min(int(h), img_h - y)

                x_centre = (x + w / 2.) / img_w
                y_centre = (y + h / 2.) / img_h
                w = float(w) / img_w
                h = float(h) / img_h

                # Limiting upto fix number of decimal places
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')

                # Writing current object
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

            file_object.close()
    def prepare_labels(self):
        print("Preparing labels...")
        if self.num_workers > 1:
            with Pool(self.num_workers) as p:
                p.starmap(self.prepare_label, zip(self.file_names))
                p.close()
                p.join()
        else:
            for filename in tqdm(self.file_names):
                self.prepare_label(filename)
        print("Done!")

    def prepare_yaml(self):
        data = {
            'path': str(self.out_path),
            "train": "train/images",
            "val": "val/images",
            "nc": 1,
            "names": ["person"]
        }
        with open(self.out_path / "dataset.yaml", "w") as f:
            yaml.dump(data, f)

class CocoToYoloMOT(CocoToYoloCrowdHuman):
    def __init__(self, data_path: Path, out_path: Path=None, split="val", image_path=None, num_workers=8):
        super().__init__(data_path, out_path, split, image_path, num_workers)
    def make_filename(self, filename):
        return filename.replace("/", '_')

    def prepare_yaml(self):
        data = {
            'path': str(self.out_path),
            "train": "train_half/images",
            "val": "val_half/images",
            "nc": 1,
            "names": ["person"]
        }
        with open(self.out_path / "dataset.yaml", "w") as f:
            yaml.dump(data, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str)
    args.add_argument('--image_path', type=str, default=None)
    args.add_argument('--split', nargs='+', default=["val", "train"])
    args.add_argument('--dataset', type=str, default="crowdhuman", help="crowdhuman or mot")
    args.add_argument('-np','--num_workers', type=int, default=8)
    args = args.parse_args()

    if args.dataset == "crowdhuman":
        coco_to_yolo = CocoToYoloCrowdHuman(Path(args.data_path), split=args.split, num_workers=args.num_workers)
    elif args.dataset == "mot":
        coco_to_yolo = CocoToYoloMOT(Path(args.data_path), split=args.split, image_path=args.image_path, num_workers=args.num_workers)
    else:
        raise NotImplementedError
    coco_to_yolo.run()