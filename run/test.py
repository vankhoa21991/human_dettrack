import cv2
from streamlit_app.helper import load_model, display_tracker_options, _display_detected_frames
import streamlit as st
from functools import partial
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import numpy as np
from copy import deepcopy
import torch


def on_predict_start(predictor, persist=False, args=None):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert args['tracking_method'] in TRACKERS, \
        f"'{args['tracking_method']}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (args['tracking_method'] + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            args['tracking_method'],
            tracking_config,
            args['reid_model'],
            predictor.device,
            args['half'],
            args['per_class']
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def track_or_det(conf, model, image, is_display_tracking=None, tracker_args=None, size=720):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    # image = cv2.resize(image, (size, int(size*9/16)))
    # Display object tracking, if specified
    if is_display_tracking:

        res = model.track(image, conf=conf,
                          show=True,
                          show_conf=True,
                          stream=False,
                          iou=0.7,
                          imgsz=[640],
                          vid_stride=1,
                          line_width=None)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    return res



def main():
    conf = 0.5
    tracker_args = {
        'half': False,
        'per_class': False,
        'tracking_method': 'ocsort',
        'reid_model': None,
        'device': '',
    }

    model_path = 'runs/detect/yolov8n_mot_ch8/weights/best.pt'
    model = load_model(model_path)

    is_display_tracker, tracker = display_tracker_options()
    model.add_callback('on_predict_start', partial(on_predict_start, persist=True, args=tracker_args))
    try:
        vid_cap = cv2.VideoCapture('data/test2.mp4')
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                res = track_or_det(conf,
                                         model,
                                         image,
                                         is_display_tracker,
                                         tracker_args,
                                         )

                # # Plot the detected objects on the video frame
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )

            else:
                vid_cap.release()
                break
    except Exception as e:
        vid_cap.release()
        raise e

def main2():
    conf = 0.5
    tracker_args = {
        'half': False,
        'per_class': False,
        'tracking_method': 'ocsort',
        'reid_model': None,
        'device': '',
    }

    model_path = 'runs/detect/yolov8n_mot_ch8/weights/best.pt'
    model = load_model(model_path)

    is_display_tracker, tracker = display_tracker_options()

    results = model.track(
        source='data/test2.mp4',
        conf=conf,
        show=True,
        stream=True,
    )
    model.add_callback('on_predict_start', partial(on_predict_start, persist=True, args=tracker_args))

    for frame_idx, r in enumerate(results):
        res_plotted = plot_res(r)
        continue



if __name__=="__main__":
    main2()