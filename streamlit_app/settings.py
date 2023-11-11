from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'data'
DEFAULT_IMAGE = IMAGES_DIR / 'test2.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'test2_pred.jpg'

# Videos config
VIDEO_DIR = ROOT / 'data'
VIDEO_1_PATH = VIDEO_DIR / 'test2.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'runs'
DETECTION_MODEL = MODEL_DIR / 'detect' / 'yolov8n_mot_ch8' / 'weights' / 'best.pt'

# Webcam
WEBCAM_PATH = 0