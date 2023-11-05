## Dataset Preparation

This page provides the instructions for dataset preparation on MOT and CrowdHuman
  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)

### 1. Download Datasets

Please download the datasets from the official websites. 

#### 1.5 Data Structure

Create your folder structure as following:

```
datasets
├── coco
│   ├── train2017
│   ├── val2017
│   ├── test2017
│   ├── annotations
│   |
├── MOT15/MOT16/MOT17/MOT20
|   ├── train
|   ├── test
│   |
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images
│   │   ├── CrowdHuman_val.zip
```

### 2. Convert Annotations
```shell
# Converto to COCO format and COCO to YOLO
# MOT17/20, please adapt the code
python datasets/mot20_to_coco.py --data_path datasets/MOT20/train/

python datasets/coco_to_yolo.py --data_path datasets/MOT20 \
                                --image_path datasets/MOT20/train \
                                --split val_half train_half --dataset mot -np 0 

# CrowdHuman
python datasets/crowdhuman_to_coco.py --data_path /home/vankhoa/datasets/crowdhuman/

python datasets/coco_to_yolo.py --data_path datasets/crowdhuman \
                                --split val train --dataset crowdhuman -np 0 
                                
# Mix dataset
python datasets/mix_mot_ch.py --paths datasets --out_path /datasets/all_data
```

The folder structure will be as following after your run these scripts:
```
datasets
├── coco
│   ├── train2017
│   ├── val2017
│   ├── test2017
│   ├── annotations
│   │
├── MOT15/MOT16/MOT17/MOT20
│   ├── train
│   ├── test
│   ├── yolo
│   │   ├── train_half
│   │   ├── val_half
│   │   ├── dataset.yaml
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images
│   │   ├── CrowdHuman_val.zip
│   ├── yolo
│   │   ├── train_half
│   │   ├── val_half
│   │   ├── dataset.yaml
├── all_data
│   ├── train
│   ├── val
│   ├── dataset.yaml
```
### 3. Mix dataset