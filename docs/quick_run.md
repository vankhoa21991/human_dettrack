## Quick Run


### Training

Run the following command to train a model:

```shell
python dettrack/detection/train_yolov8.py  ${CONFIG_FILE}

# Example
python dettrack/detection/train_yolov8.py -cfg config/yolov8n_mot_ch.yaml
```

### Validation
Benchmark result on MOT17 dataset:

```shell
python run/val.py --data_path /home/vankhoa/datasets \
                  --yolo-model runs/detect/yolov8n_mot_ch8/weights/best.pt \
                  --tracking-method deepocsort \
                  --benchmark MOT17 \
                  --processes-per-device 4 \
                  --split val 
```
### Inference

##### Simple Tracking Demo

1. Configuration: Modify the configuration file config.yaml to set your preferred detection and tracking algorithms, camera sources, and other parameters.
2. Run the Application:

```shell
python run_tracker_simple.py
```
#### Detection and Tracking Demo

```shell
python run/track.py --yolo-model runs/detect/yolov8s_crowdhuman13/weights/best.pt \
                    --tracking-method ocsort \
                    --source test2.mp4 \
                    --show \
                    --save
```


