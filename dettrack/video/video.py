import cv2
from pathlib import Path

class videoHuman():
    def __init__(self, cfg):
        self.scale_percent = cfg['scale_percent']

    def set_source_video(self, source):
        self.capture = cv2.VideoCapture(source)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def set_source_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def print_size(self):
        print('width, height:', self.width, self.height)

    def scale(self):
        # Scaling Video for better performance
        if self.scale_percent != 100:
            print('[INFO] - Scaling change may cause errors in pixels lines ')
            self.width = int(self.width * self.scale_percent / 100)
            self.width = int(self.height * self.scale_percent / 100)
            print('[INFO] - Dim Scaled: ', (self.width, self.height))
            return self.width, self.height
        else:
            return self.width, self.height

    def get_writer(self, output_path:Path):
        self.VIDEO_CODEC = "mp4v"

        writer = cv2.VideoWriter( str(output_path),
                                       cv2.VideoWriter_fourcc(*self.VIDEO_CODEC),
                                       self.fps, (self.width, self.height))
        return writer


