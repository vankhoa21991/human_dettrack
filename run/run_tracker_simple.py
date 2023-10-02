import cv2
from ultralytics import YOLO
#plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import subprocess
from tqdm import tqdm
import urllib.request
import shutil
import argparse
import yaml
from dettrack.utils.utils import risize_frame, filter_tracks, update_tracking
from dettrack.detection.detector_yolov8 import detectorYoloV8
from dettrack.video.video import videoHuman
from dettrack.tracking.tracker_simple import Tracker
def main(args, cfg, verbose=False):
    print(f'[INFO] - Verbose during Prediction: {verbose}')
    scale_percent = cfg['scale_percent']
    patience = cfg['patience']

    video = videoHuman(cfg)
    if args.video:
        video.set_source_video(args.video)
    else:
        video.set_source_webcam()
    video.print_size()
    video.scale()
    capture = video.capture

    detector = detectorYoloV8(cfg)
    tracker = Tracker(cfg)

    centers_old = {}
    count_p = 0
    lastKey = ''

    writer = video.get_writer(args.output)

    i=0
    while True:

        # reading frame from video
        _, frame = capture.read()

        # Applying resizing of read frame
        frame = risize_frame(frame, scale_percent)

        ROI = frame

        # Getting predictions
        y_hat = detector.detect(ROI)

        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        conf = y_hat[0].boxes.conf.cpu().numpy()

        # Storing the above information in a dataframe
        positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes,
                                       columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf',
                                                'class'])  # this will be deprecated in the future version of ultralytics

        # For each people, draw the bounding-box and counting each one the pass thought the ROI area
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')

            # Calculating the center of the bounding-box
            obj_center = (int(((xmax + xmin)) / 2), int((ymax + ymin) / 2))

            centers_old, id_obj, is_new, lastKey = tracker.update(centers_old, obj_center, lastKey, i)

            # Updating people in roi
            count_p += is_new

            # drawing center and bounding-box in the given frame
            cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # box
            for center_x, center_y in centers_old[id_obj].values():
                cv2.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)  # center of box

            # Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)) + ' ' + str(detector.dict_classes[category]),
                        org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                        thickness=1)

        # drawing the number of people
        cv2.putText(img=frame, text=f'Counts People in ROI: {count_p}',
                    org=(30, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=1)

        # Filtering tracks history
        centers_old = tracker.filter_tracks(centers_old, patience)

        cv2.imshow("yolov8", frame)

        # saving transformed frames in a output video formaat
        writer.write(frame)
        i += 1

        if (cv2.waitKey(30) == 27):  # break with escape key
            break

    # Releasing the video
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-cfg", "--config", type=str, default="config/yolov8x.yaml", help="path to model config file")
    args.add_argument("--video", type=str, default=None, help="path to video file")
    args.add_argument("--output", type=str, default="output.mp4", help="path to output video file")
    args = args.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(args, cfg)