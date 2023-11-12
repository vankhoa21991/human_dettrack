from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import streamlit_app.settings as settings
import cv2
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
            None,
            predictor.device,
            False,
            False
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    assert Path(model_path).exists(), 'Model file does not exist.'
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack", "ocsort"))
        print(tracker_type)
        return is_display_tracker, tracker_type
    return is_display_tracker, None


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
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, classes=[0])
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res.plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


# def play_youtube_video(conf, model):
#     """
#     Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.
#
#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#
#     Returns:
#         None
#
#     Raises:
#         None
#     """
#     source_youtube = st.sidebar.text_input("YouTube Video url")
#
#     is_display_tracker, tracker = display_tracker_options()
#
#     if st.sidebar.button('Detect Objects'):
#         try:
#             yt = YouTube(source_youtube)
#             stream = yt.streams.filter(file_extension="mp4", res=720).first()
#             vid_cap = cv2.VideoCapture(stream.url)
#
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker,
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))


# def play_rtsp_stream(conf, model):
#     """
#     Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.
#
#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#
#     Returns:
#         None
#
#     Raises:
#         None
#     """
#     source_rtsp = st.sidebar.text_input("rtsp stream url:")
#     st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
#     is_display_tracker, tracker = display_tracker_options()
#     if st.sidebar.button('Detect Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(source_rtsp)
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker
#                                              )
#                 else:
#                     vid_cap.release()
#                     # vid_cap = cv2.VideoCapture(source_rtsp)
#                     # time.sleep(0.1)
#                     # continue
#                     break
#         except Exception as e:
#             vid_cap.release()
#             st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """

    is_display_tracker, tracker = display_tracker_options()
    results = model.track(
        source=settings.WEBCAM_PATH,
        conf=conf,
        show=False,
        stream=True,
        classes=[0]
    )
    model.add_callback('on_predict_start', partial(on_predict_start, persist=True, args={'tracking_method': tracker}))
    if st.sidebar.button('Detect Objects'):
        try:
            # vid_cap = cv2.VideoCapture(settings.WEBCAM_PATH)
            # st_frame = st.empty()
            # while (vid_cap.isOpened()):
            #     success, image = vid_cap.read()
            #     if success:
            #         _display_detected_frames(conf,
            #                                  model,
            #                                  st_frame,
            #                                  image,
            #                                  is_display_tracker,
            #                                  tracker,
            #                                  )
            #     else:
            #         vid_cap.release()
            #         break

            st_frame = st.empty()
            for frame_idx, r in enumerate(results):
                res_plotted = plot_res(r)
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )


        except Exception as e:
            # vid_cap.release()
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()
    results = model.track(
        source=str(settings.VIDEOS_DICT.get(source_vid)),
        conf=conf,
        show=False,
        stream=True,
        classes=[0]
    )
    model.add_callback('on_predict_start', partial(on_predict_start, persist=True, args={'tracking_method': tracker}))
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            st_frame = st.empty()
            for frame_idx, r in enumerate(results):
                res_plotted = plot_res(r)
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                continue
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def plot_res(res, conf=True,
            line_width=None,
            font_size=None,
            font='Arial.ttf',
            pil=False,
            img=None,
            im_gpu=None,
            kpt_radius=5,
            kpt_line=True,
            labels=True,
            boxes=True,
            masks=True,
            probs=True,
             **kwargs):

    if img is None and isinstance(res.orig_img, torch.Tensor):
        img = np.ascontiguousarray(res.orig_img[0].permute(1, 2, 0).cpu().detach().numpy()) * 255

    names = res.names
    pred_boxes, show_boxes = res.boxes, boxes
    pred_masks, show_masks = res.masks, masks
    pred_probs, show_probs = res.probs, probs
    annotator = Annotator(
        deepcopy(res.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names)

    # Plot Detect results
    if pred_boxes and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ('' if id is None else f'id:{id} ') + names[c]
            label = (f'{name} {conf:.2f}' if conf else name) if labels else None
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
        x = round(res.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    # Plot Pose results
    if res.keypoints is not None:
        for k in reversed(res.keypoints.data):
            annotator.kpts(k, res.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

    return annotator.result()