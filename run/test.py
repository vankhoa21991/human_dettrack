import cv2
from streamlit_app.helper import load_model, display_tracker_options, _display_detected_frames
import streamlit as st
from functools import partial
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT
from pathlib import Path

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

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
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
    image = cv2.resize(image, (720, int(720)))

    # Display object tracking, if specified
    if is_display_tracking:
        args = {
            'half': False,
            'per_class': False,
            'tracking_method': 'ocsort',
            'reid_model': Path('weights') / 'osnet_x0_25_msmt17.pt',
            'device': '',
        }
        model.add_callback('on_predict_start', partial(on_predict_start, persist=True, args=args))
        res = model.track(image, conf=conf, persist=True, tracker=tracker, show=True)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def main():
    conf = 0.4
    model_path = 'runs/detect/yolov8n_mot_ch8/weights/best.pt'
    model = load_model(model_path)

    is_display_tracker, tracker = display_tracker_options()
    try:
        vid_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf,
                                         model,
                                         st_frame,
                                         image,
                                         is_display_tracker,
                                         tracker,
                                         )
            else:
                vid_cap.release()
                break
    except Exception as e:
        vid_cap.release()
        raise e

if __name__=="__main__":
    main()