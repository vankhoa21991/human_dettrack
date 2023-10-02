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

def run_on_webcam(args, cfg, verbose=False):
    scale_percent = cfg['scale_percent']
    conf_level = cfg['conf_level']
    thr_centers = cfg['thr_centers']
    patience = cfg['patience']
    alpha = cfg['alpha']
    frame_max = cfg['frame_max']

    # loading a YOLO model
    model = YOLO(cfg['model'])

    # geting names from classes
    dict_classes = model.model.names

    # Reading video with cv2
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Objects to detect Yolo
    class_IDS = [0]
    # Auxiliary variables
    centers_old = {}

    obj_id = 0
    end = []
    frames_list = []
    count_p = 0
    lastKey = ''
    print(f'[INFO] - Verbose during Prediction: {verbose}')

    # Original informations of video
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim: ', (width, height))

    # Scaling Video for better performance
    if scale_percent != 100:
        print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled: ', (width, height))

    # -------------------------------------------------------
    ### Video output ####
    video_name = 'result.mp4'
    output_path = "rep_" + video_name
    tmp_output_path = "tmp_" + output_path
    VIDEO_CODEC = "mp4v"

    output_video = cv2.VideoWriter(tmp_output_path,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))

    # -------------------------------------------------------
    # Executing Recognition
    i=0
    while True:

        # reading frame from video
        _, frame = video.read()

        # Applying resizing of read frame
        frame = risize_frame(frame, scale_percent)
        #     frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        area_roi = [np.array([(1250, 400), (750, 400), (700, 800), (1200, 800)], np.int32)]
        ROI = frame[0:2000, 0:2000]

        # Getting predictions
        y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device=0, verbose=False)

        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()
        conf = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy()

        # Storing the above information in a dataframe
        positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes,
                                       columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf',
                                                'class'])  # this will be deprecated in the future version of ultralytics

        # Translating the numeric class labels to text
        labels = [dict_classes[i] for i in classes]

        # For each people, draw the bounding-box and counting each one the pass thought the ROI area
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')

            # Calculating the center of the bounding-box
            center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

            # Updating the tracking for each object
            centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y), thr_centers,
                                                                   lastKey, i, frame_max)

            # Updating people in roi
            count_p += is_new

            # drawing center and bounding-box in the given frame
            cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # box
            for center_x, center_y in centers_old[id_obj].values():
                cv2.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)  # center of box

            # Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)),
                        org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                        thickness=1)

        # drawing the number of people
        cv2.putText(img=frame, text=f'Counts People in ROI: {count_p}',
                    org=(30, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1.5, color=(255, 0, 0), thickness=1)

        # Filtering tracks history
        centers_old = filter_tracks(centers_old, patience)

        cv2.imshow("yolov8", frame)

        # Drawing the ROI area
        overlay = frame.copy()

        cv2.polylines(overlay, pts=area_roi, isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.fillPoly(overlay, area_roi, (255, 0, 0))
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # saving transformed frames in a output video formaat
        output_video.write(frame)
        i += 1

        if (cv2.waitKey(30) == 27):  # break with escape key
            break

    # Releasing the video
    output_video.release()
    cv2.destroyAllWindows()

def main(args, cfg, verbose=False):
    scale_percent = cfg['scale_percent']
    conf_level = cfg['conf_level']
    thr_centers = cfg['thr_centers']
    patience = cfg['patience']
    alpha = cfg['alpha']
    frame_max = cfg['frame_max']

    # loading a YOLO model
    model = YOLO(cfg['model'])

    # geting names from classes
    dict_classes = model.model.names

    # Reading video with cv2
    video = cv2.VideoCapture(args.video)


    # Objects to detect Yolo
    class_IDS = [0]
    # Auxiliary variables
    centers_old = {}

    obj_id = 0
    end = []
    frames_list = []
    count_p = 0
    lastKey = ''
    print(f'[INFO] - Verbose during Prediction: {verbose}')

    # Original informations of video
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim: ', (width, height))

    # Scaling Video for better performance
    if scale_percent != 100:
        print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled: ', (width, height))

    # -------------------------------------------------------
    ### Video output ####
    video_name = 'result.mp4'
    output_path = "rep_" + video_name
    tmp_output_path = "tmp_" + output_path
    VIDEO_CODEC = "mp4v"

    output_video = cv2.VideoWriter(tmp_output_path,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))

    # -------------------------------------------------------
    # Executing Recognition
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):

        # reading frame from video
        _, frame = video.read()

        # Applying resizing of read frame
        frame = risize_frame(frame, scale_percent)
        #     frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        area_roi = [np.array([(1250, 400), (750, 400), (700, 800), (1200, 800)], np.int32)]
        ROI = frame[0:2000, 0:2000]

        if verbose:
            print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

        # Getting predictions
        y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device=0, verbose=False)

        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()
        conf = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy()

        # Storing the above information in a dataframe
        positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes,
                                       columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']) # this will be deprecated in the future version of ultralytics

        # Translating the numeric class labels to text
        labels = [dict_classes[i] for i in classes]

        # For each people, draw the bounding-box and counting each one the pass thought the ROI area
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')

            # Calculating the center of the bounding-box
            center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

            # Updating the tracking for each object
            centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y), thr_centers,
                                                                   lastKey, i, frame_max)

            # Updating people in roi
            count_p += is_new

            # drawing center and bounding-box in the given frame
            cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # box
            for center_x, center_y in centers_old[id_obj].values():
                cv2.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)  # center of box

            # Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)),
                        org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                        thickness=1)

        # drawing the number of people
        cv2.putText(img=frame, text=f'Counts People in ROI: {count_p}',
                    org=(30, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1.5, color=(255, 0, 0), thickness=1)

        # Filtering tracks history
        centers_old = filter_tracks(centers_old, patience)

        # Drawing the ROI area
        overlay = frame.copy()

        #cv2.polylines(overlay, pts=area_roi, isClosed=True, color=(255, 0, 0), thickness=2)
        #cv2.fillPoly(overlay, area_roi, (255, 0, 0))
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Saving frames in a list
        frames_list.append(frame)
        # saving transformed frames in a output video formaat
        output_video.write(frame)

    # Releasing the video
    output_video.release()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-cfg", "--config", type=str, default="config/yolov8x.yaml", help="path to model config file")
    args.add_argument("--video", type=str, default="data/vid1.mp4", help="path to video file")
    args.add_argument("--output", type=str, default="output.mp4", help="path to output video file")
    args = args.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    #main(args, cfg)
    run_on_webcam(args, cfg)